##  Miner Setup Guide

To contribute as a Miner in the NIOME(SN55) and earn **TAO (**$\tau$**)** emissions, you should first set up your machine, install the necessary packages, and register your identity on the Bittensor.

### 1. Prerequisites

Before starting, ensure your system meets the minimum requirements and has the core dependencies installed.

* **Operating System:** Ubuntu 22.04 or similar Linux distribution is generally recommended for optimal compatibility. Mining is not supported on Windows.
* **Python:** Python 3.12
* **Git:** cloning the repository
* **Hardware:** 
   - vCPU : 16
   - GPU : necessary, up to your generation
   - Memory : 16GB minimum
   - 3rd Party API : unnecessary
   - Port Forwarding : standard

### 2. Environment Setup

This section walks you through cloning the subnet-niome repository and installing the required packages.

1. **Clone the Repository:**
   **Bash**

   ```
   git clone https://github.com/genomesio/subnet-niome.git
   cd subnet-niome
   ```
2. **Create a Virtual Environment (Recommended):**
   **Bash**

   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies:** Install the required Python packages and register the local package for execution.
   **Bash**

   ```
   python3 -m pip install -r requirements.txt
   ```

### 3. Wallet Creation and Registration

You must create a Bittensor wallet to hold your TAO and Alpha tokens, and to register your hotkey with the subnet.

1. **Install Bittensor CLI:**
   **Bash**

   ```
   python3 -m pip install bittensor-cli
   ```
2. **Create a Coldkey (Primary Wallet):** The coldkey is your secure, offline store of funds. Choose a secure name
   **Bash**

   ```
   btcli wallet new_coldkey --wallet.name your_coldkey
   ```
3. **Create a Hotkey (Miner Identity):** The hotkey is used to sign transactions, run the miner, and receive emissions. It is connected to your coldkey.
   **Bash**

   ```
   btcli wallet new_hotkey --wallet.name your_coldkey --wallet.hotkey your_hotkey
   ```
4. **Fund Your Coldkey:** Transfer a small amount of TAO to your coldkey to cover registration fees, which fluctuate based on subnet competition.
5. **Register Your Hotkey to Subnet 55:** Register your hotkey to secure a UID (Unique Identifier) on the NIOME subnet. The Network ID for NIOME is 55.
   **Bash**

   ```
   btcli subnet register --netuid 55 --wallet.name your_coldkey --wallet.hotkey your_hotkey
   ```

### 4. Running the Miner

Once your hotkey is registered, you can start your Miner. 

1. **Run the Miner Script:** The core command to launch a miner neuron requires specifying your wallet and hotkey names, the network, and the subnet ID (`--netuid 55`).
   
   In your current subnet-niome project path

   **Bash**
   ```
   export PYTHONPATH="$PYTHONPATH:$(pwd)                                                               
   ```

   ```
   python neurons/miner.py \
   --netuid 55 \
   --subtensor.network finney \
   --wallet.name your_coldkey \
   --wallet.hotkey your_hotkey \
   --axon.port your_port
   ```

3. **Keep it Running:** Use a process manager like **`pm2`** or **`tmux`** to ensure your miner remains active and online, as Validators reward only active, responsive miners.