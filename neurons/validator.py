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


import time

# Bittensor
import bittensor as bt

import niome_subnet
import wandb

# import base validator class which takes care of most of the boilerplate
from niome_subnet.base.validator import BaseValidatorNeuron
from niome_subnet.utils import TESTNET_UID
from niome_subnet.validator import forward

bt.logging.on()


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        self.load_state()
        
        self.init_wandb()

    def build_signature_headers(self, signature, hotkey, timestamp, netuid) -> dict:
        """
        Build headers for validator requests.

        Args:
            signature (str): Signature of the request
            hotkey (str): Hotkey of the validator
            timestamp (float): Timestamp of the request
            netuid (str): Netuid of the subnet
        Returns:
            dict: Headers for the request
        """
        return {
            "X-Signature": signature,
            "X-Hotkey": hotkey,
            "X-Netuid": netuid,
            "X-Timestamp": timestamp,
        }

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        return await forward(self)
    
    def init_wandb(self):
        if self.config.wandb.off:
            return
        
        run_name = f"validator-{self.uid}-{niome_subnet.__version__}"
        self.config.run_name = run_name
        self.config.uid = self.uid
        self.config.hotkey = self.wallet.hotkey.ss58_address
        self.config.version = niome_subnet.__version__
        self.config.type = self.neuron_type

        wandb_project = (
            self.config.wandb.testnet_project_name
            if self.config.netuid == TESTNET_UID
            else self.config.wandb.project_name
        )

        bt.logging.info(
            f"Initializing W&B run for '{self.config.wandb.entity}/{wandb_project}'"
        )
        try:
            wandb.login(key=self.config.wandb.api_key, relogin=True)
            run_id = wandb.init(
                name=run_name,
                project=wandb_project,
                entity=self.config.wandb.entity,
                config=self.config,
                dir=self.config.full_path,
                mode="offline" if self.config.wandb.offline else None
            ).id
        except wandb.UsageError as e:
            bt.logging.warning(e)
            bt.logging.warning("Did you run  wandb login?")
            return

        signature = self.wallet.hotkey.sign(run_id.encode()).hex()
        self.config.signature = signature
        wandb.config.update(self.config, allow_val_change=True)

        bt.logging.success(f"Started wandb run {run_name}")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            # bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
