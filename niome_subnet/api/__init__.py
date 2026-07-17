import bittensor as bt
import boto3
import json
import niome_subnet.utils.settings as config
import requests
import time
import urllib.request

from niome_subnet.genomics.model import MinerScoreDto, Task


def get(self, url: str):
    timestamp = str(time.time())
    canonical = json.dumps({
        'payload': json.dumps({}, separators=(',', ':'), sort_keys=True),
        'hotkey': self.wallet.hotkey.ss58_address,
        'netuid': str(self.netuid),
        'timestamp': timestamp,
    }, separators=(',', ':'), sort_keys=True)

    signature = self.wallet.hotkey.sign(canonical).hex()

    for attempt in range(1, config.MAX_TASK_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=self.build_signature_headers(
                    signature=signature,
                    hotkey=self.wallet.hotkey.ss58_address,
                    timestamp=timestamp,
                    netuid=str(self.netuid),
                ),
                timeout=config.TASK_REQUEST_TIMEOUT,
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Backend returned status {response.status_code}"
                )

            return response.json()
        except Exception as e:
            bt.logging.error(f"Get Error: {str(e)}")
            if attempt < config.MAX_TASK_RETRIES:
                delay = config.BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                bt.logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                bt.logging.error("All retries failed")
                raise e

def post(self, url: str, payload: dict):
    timestamp = str(time.time())
    canonical = json.dumps({
        'payload': json.dumps(payload, separators=(',', ':'), sort_keys=True),
        'hotkey': self.wallet.hotkey.ss58_address,
        'netuid': str(self.netuid),
        'timestamp': timestamp,
    }, separators=(',', ':'), sort_keys=True)

    signature = self.wallet.hotkey.sign(canonical).hex()

    for attempt in range(1, config.MAX_TASK_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers=self.build_signature_headers(
                    signature=signature,
                    hotkey=self.wallet.hotkey.ss58_address,
                    timestamp=timestamp,
                    netuid=str(self.netuid),
                ),
                json=payload,
                timeout=config.TASK_REQUEST_TIMEOUT,
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Backend returned status {response.status_code}"
                )

            return response.json()
        except Exception as e:
            bt.logging.error(f"Post Error: {e}")
            if attempt < config.MAX_TASK_RETRIES:
                delay = config.BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                bt.logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                bt.logging.error("All retries failed")
                raise e

def fetch_task(self) -> Task:
    """Generate a synthetic genomic simulation task with retry logic and fallback."""
    data = get(self, config.TASK_URL)
    
    # API returns task_id but model expects id
    if "task_id" in data:
        data["id"] = data.pop("task_id")
    
    task = Task.model_validate(data)

    if not task.contract_url or not task.hbb_ref_url:
        bt.logging.error("Invalid response: missing contract_url or hbb_ref_url")
        raise RuntimeError("Invalid response")

    urllib.request.urlretrieve(task.contract_url, config.CONTRACT_PATH)
    urllib.request.urlretrieve(task.hbb_ref_url, config.HBB_REFERENCE_PATH)

    return task

def submit_validation_result(self, miner_scores: list[MinerScoreDto]) -> None:
    """Submit miner scores with retry logic and fallback."""
    payload = {
      "scores": [score.model_dump() for score in miner_scores],
    }
    post(self, config.MINER_SCORE_URL, payload)

def upload_final_submission_to_server(self, uids: list[int]) -> None:
    """Download submissions by UIDs from S3, merge them, and upload to server."""
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION,
        )
        
        merged_submissions = []
        
        # Download and merge submissions from S3
        for uid in uids:
            try:
                # Construct S3 key for the submission file
                s3_key = f"niome/{uid}.json"
                local_path = f"data/temp_submission.json"
                
                # Download from S3
                s3_client.download_file(
                    config.AWS_S3_BUCKET,
                    s3_key,
                    local_path
                )
                
                # Load and merge the submission
                with open(local_path, "r") as f:
                    submission_data = json.load(f)
                    
                    # If submission is a list, extend merged submissions
                    if isinstance(submission_data, list):
                        merged_submissions.extend(submission_data)
                    # If submission is a dict, append it
                    else:
                        merged_submissions.append(submission_data)
            except Exception as e:
                bt.logging.error(f"Error downloading submission for UID {uid}: {e}")
                continue
        
        if not merged_submissions:
            bt.logging.warning("No submissions to upload")
            return
        
        # Upload merged submissions to server
        bt.logging.info(f"Uploading {len(merged_submissions)} merged submissions to server")
        payload = {
            "task_id": self.task_id,
            "submission": merged_submissions,
        }
        
        post(self, config.MINER_SUBMISSION_URL, payload)
        bt.logging.info(f"Successfully uploaded merged submissions")
        
    except Exception as e:
        bt.logging.error(f"Error in upload_submissions_to_server: {e}")
        raise e
