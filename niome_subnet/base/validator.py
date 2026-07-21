# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 genomes.io

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import argparse
import bittensor as bt
import copy
import numpy as np
import threading

from datetime import datetime, timezone
from typing import List, Union
from traceback import print_exception

from niome_subnet.api import submit_validation_result
from niome_subnet.base.neuron import BaseNeuron
from niome_subnet.genomics.model import MinerScore, MinerScoreDto
from niome_subnet.utils import (
    add_validator_args,
    convert_weights_and_uids_for_emit,
    process_scores_linear,
    process_scores_top,
    process_weights_for_netuid,
)
from niome_subnet.mock import MockDendrite

from niome_subnet.utils.settings import BURNING_RATE, OWNER_HOTKEY, SCORING_SYSTEM


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.Dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        # Ensure metagraph.n is a Python int to avoid numpy's __mul__ dispatch
        # when multiplying lists by numpy scalar types.
        self.file_names = np.array([""] * int(self.metagraph.n), dtype=object)
        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()
        self.is_broadcasting: bool = False
        self.is_validating: bool = False
        self.task_id: str = ""

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.Axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                # bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                if self.should_set_weights():
                    self.load_state()
                    result, msg = self.subtensor.set_weights(
                        wallet=self.wallet,
                        netuid=self.config.netuid,
                        uids=self.uids,
                        weights=self.weights,
                        wait_for_finalization=False,
                        wait_for_inclusion=False,
                    )
                    if result:
                        bt.logging.info("set_weights on chain successfully!")
                    else:
                        bt.logging.error("set_weights failed", msg)

                    self.uids = []
                    self.weights = []

                # Sync metagraph.
                self.sync()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.debug(str(print_exception(type(err), err, err.__traceback__)))

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def build_signed_headers(self) -> dict:
        timestamp = int(datetime.now(tz=timezone.utc).timestamp())
        message = f"<Signature>{timestamp}</Signature>"
        signature = self.wallet.hotkey.sign(message)
        return {
            "X-Validator-Hotkey": self.wallet.hotkey.ss58_address,
            "X-Validator-Signature": signature.hex(),
            "X-Validator-Timestamp": str(timestamp),
        }

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self, scores: List[MinerScore], task_id: str):
        """
        Sets the validator weights on-chain.
        Miners receive total weight * (1 - BURNING_RATE)
        OWNER_HOTKEY receives the remaining BURNING_RATE
        """
        owner_uid = self.metagraph.hotkeys.index(OWNER_HOTKEY)

        scores_array = np.zeros(self.metagraph.n, dtype=np.float32)
        for ms in scores:
            if 0 <= ms.uid < self.metagraph.n:
                scores_array[ms.uid] = ms.final_score
        scores_array = np.nan_to_num(scores_array)

        # process_scores_* already return a normalized weight vector summing to 1.0
        if SCORING_SYSTEM == "linear":
            processed_scores = process_scores_linear(scores_array)
        else:
            processed_scores = process_scores_top(scores_array)

        processed_weight_uids, processed_weights = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=processed_scores,
            netuid=self.config.netuid,
            owner_uid=owner_uid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        full_miner_weights = np.zeros(self.metagraph.n, dtype=np.float32)
        if len(processed_weight_uids) > 0:
            full_miner_weights[np.asarray(processed_weight_uids)] = processed_weights

        full_sum = np.sum(full_miner_weights)
        if full_sum > 0:
            full_miner_weights /= full_sum

        final_weights = full_miner_weights * (1 - BURNING_RATE)

        if 0 <= owner_uid < self.metagraph.n:
            final_weights[owner_uid] += BURNING_RATE
        else:
            bt.logging.warning(f"OWNER_UID {owner_uid} out of range!")

        final_norm = np.sum(final_weights) or 1.0
        final_weights /= final_norm

        bt.logging.info(f"Weights: {final_weights}")

        miner_score_dtos: list[MinerScoreDto] = [
            MinerScoreDto(
                task_id=task_id,
                uid=ms.uid,
                hotkey=self.metagraph.hotkeys[ms.uid],
                stage1=ms.stage1.mean * (ms.stage1.mean * ms.stage1.valid) if ms.stage1 else 0.0,
                stage2=ms.stage2.mean if ms.stage2 else 0.0,
                stage3=ms.stage3.mean_energy if ms.stage3 else 0.0,
                stage4=ms.stage4.consistency_score if ms.stage4 else 0.0,
                final_score=ms.final_score,
                log=ms.log,
                weight=float(final_weights[ms.uid]),
            )
            for ms in scores
            if 0 <= ms.uid < self.metagraph.n
        ]

        if task_id != "":
            submit_validation_result(self, miner_scores=miner_score_dtos)
            bt.logging.info(f"Submitted validation results for task {task_id} to server.")

        final_uids = np.where(final_weights > 1e-8)[0].tolist()
        final_weight_values = final_weights[final_uids].tolist()

        self.uids, self.weights = convert_weights_and_uids_for_emit(
            uids=final_uids, weights=final_weight_values
        )
        self.save_state()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        for uid, hotkey in enumerate(self.hotkeys):
            if uid < len(self.metagraph.hotkeys) and hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0
                self.file_names[uid] = ""

        if len(self.scores) != self.metagraph.n:
            new_scores = np.zeros(self.metagraph.n, dtype=np.float32)
            new_file_names = np.array([""] * int(self.metagraph.n), dtype=object)
            min_len = min(len(self.scores), self.metagraph.n)
            new_scores[:min_len] = self.scores[:min_len]
            new_file_names[:min_len] = self.file_names[:min_len]
            self.scores = new_scores
            self.file_names = new_file_names

        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def save_state(self):
        """Saves the state of the validator to a file."""
        # Save the state of the validator to file.
        try:
            np.savez(
                self.config.neuron.full_path + "/state.npz",
                step=self.step,
                uids=self.uids,
                weights=self.weights,
                validated_uids=self.validated_uids,
                task_id=self.task_id,
            )
        except Exception as e:
            bt.logging.error(f"Failed to save state with exception: {e}")

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")
        try:
            state = np.load(self.config.neuron.full_path + "/state.npz")
            if isinstance(state["step"], (int, np.integer, np.ndarray)):
                self.step = int(state["step"])
            if isinstance(state["uids"], np.ndarray):
                self.uids = state["uids"].tolist()
            if isinstance(state["weights"], np.ndarray):
                self.weights = state["weights"].tolist()
            if "task_id" in state:
                self.task_id = str(state["task_id"])
            if "validated_uids" in state:
                self.validated_uids = state["validated_uids"].tolist()
        except Exception as e:
            bt.logging.error(f"Failed to load state with exception: {e}")
