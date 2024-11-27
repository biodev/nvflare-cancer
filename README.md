# NVFlare in Cancer Research 

[![DOI](https://zenodo.org/badge/794854623.svg)](https://zenodo.org/doi/10.5281/zenodo.13124379)

We are setting up a project among multiple cancer research organizations to explore federated learning using NVFlare. This is a pilot co-led by OHSU Knight Cancer Institute and NCI. We assume that the participants will have one of these options available:

- An AWS or Azure cloud account 
- An on-premises Slurm-based HPC system with GPU nodes
- A Windows laptop with a GPU and WSL Linux installed 

The central NVFlare dashboard and server was installed by the `Project Admin`. An individual with this role is responsible for setting up the project and approving requesting sites and users. This person should be a senior IT/IPS professional from one of the collaborating sites. Individuals with this role are in charge of server security and by default have unrestricted access to the central NVFlare system which is installed in a hub-and-spoke model. The researchers in all participating institutions will use an NVFlare compute client on their HPC system, their laptop or a separate cloud account and will have no access to the central hub while the `Project Admin` of the central hub will have no access to the data processd by the clients. Please review the [terminologies and roles](https://nvflare.readthedocs.io/en/main/user_guide/security/terminologies_and_roles.html) required for a funtioning NVFlare federation.


# Table of Contents

- [NVFlare in Cancer Research](#nvflare-in-cancer-research)
- [Table of Contents](#table-of-contents)
- [Installing NVFlare deploy environment](#installing-nvflare-deploy-environment)
  - [Prerequisites](#prerequisites)
  - [Installing the right version of Python](#installing-the-right-version-of-python)
  - [Installing NVFlare in an isolated venv](#installing-nvflare-in-an-isolated-venv)
- [Connecting to an existing NVFlare Project](#connecting-to-an-existing-nvflare-project)
  - [Using NVFlare as a Lead](#using-nvflare-as-a-lead)
    - [Using the administrative console](#using-the-administrative-console)
    - [Testing a Python example](#testing-a-python-example)
    - [Testing a Pytorch example](#testing-a-pytorch-example)
      - [Pytorch simulator mode](#pytorch-simulator-mode)
      - [Pytorch poc mode](#pytorch-poc-mode)
    - [Client API with external process (Slurm use case)](#client-api-with-external-process-slurm-use-case)
      - [Connecting to HPC with an older version of NVFlare](#connecting-to-hpc-with-an-older-version-of-nvflare)
    - [Troubleshooting](#troubleshooting)
      - [Missing server dependencies](#missing-server-dependencies)
      - [No default VPC](#no-default-vpc)
      - [SSL issue](#ssl-issue)
  - [Using NVFlare as an Org Admin](#using-nvflare-as-an-org-admin)
    - [Register client sites](#register-client-sites)
      - [Enter available GPU memory](#enter-available-gpu-memory)
    - [Install a client on AWS](#install-a-client-on-aws)
      - [Upgrade all packages](#upgrade-all-packages)
    - [Install a client on HPC](#install-a-client-on-hpc)
      - [Install a client on the HPC login node](#install-a-client-on-the-hpc-login-node)
    - [Install a client on your WSL Laptop](#install-a-client-on-your-wsl-laptop)
    - [Verify installation](#verify-installation)
- [(Not needed right now): Deploying a new NVFlare Project](#not-needed-right-now-deploying-a-new-nvflare-project)
  - [Installing Dashboard](#installing-dashboard)
    - [Getting dashboard production ready](#getting-dashboard-production-ready)
      - [About 1. Start at reboot](#about-1-start-at-reboot)
      - [About 2. Restart on Error](#about-2-restart-on-error)
      - [About 3. DNS/HTTPS](#about-3-dnshttps)
        - [DNS and SSL Certificate through your IT department](#dns-and-ssl-certificate-through-your-it-department)
        - [DNS through Route53 and let's encrypt SSL](#dns-through-route53-and-lets-encrypt-ssl)
    - [Configuring Dashboard](#configuring-dashboard)
    - [Resetting Dashboard](#resetting-dashboard)
  - [Installing Server](#installing-server)
  - [Installing Client](#installing-client)
- [Contributing code to NVFlare](#contributing-code-to-nvflare)

# Installing NVFlare deploy environment 

First we install the NVFlare software packages on a computer from which you will connect to the infrastructure and/or roll it out in the first place. This can be a laptop or management server.

## Prerequisites

If you want to roll out parts of the infrastruncture to AWS or Azure, you should have the AWS or Azure CLI installed and AWS or Azure credentials setup. You must be allowed to launch AWS EC2 instances or Azure virtual machines including network configuration.

## Installing the right version of Python

For consistency reasons we recommend installing the latest Python version supported by NVFlare (NVFlare 2.5.1 and Python 3.12 as of October 2024). For our AWS deployment we will use Ubuntu 24.04, which comes with Python 3.12. To quickly install Python 3.12 in your work environment (Linux, Mac or Windows with WSL Linux) we propose the Rye Package manager by Armin Ronacher (the maker of Flask) as it very fast and can be easily removed. Below are the instructions for Linux (incl. WSL) and Mac. Do not use the Windows instructions [here](https://rye.astral.sh/) as they are not tested. Run and accept the defaults:  

```bash
curl -sSf https://rye.astral.sh/get | bash
```

Rye quickly installs Python 3.12 in a reproducible way and makes it the default Python on your system (it will edit file ~/.python-version)

```
. ~/.rye/env
rye fetch 3.12
rye pin 3.12
```

The Rye installer will put `. ~/.rye/env` into ~/.profile to ensure that this Python version is ready at the next login. If you have an older HPC system it may not use the file ~/.profile but ~/.bash_profile instead. just run `echo '. ~/.rye/env' >> ~/.bash_profile` in that case. 

A quick test should show, that the default python is latest Python 3.12

```bash
$ python --version
Python 3.12.3
```

## Installing NVFlare in an isolated venv

install NVFlare in a new virtual environment at `~/.local/nvf` and source it

```bash
$ rye init ~/.local/nvf && cd ~/.local/nvf && rye add nvflare jupyterlab pip && rye sync && cd ~

success: Initialized project in /home/pytester/.local/nvf
  Run `rye sync` to get started
Initializing new virtualenv in /home/pytester/.local/nvf/.venv
Python version: cpython@3.12.5
Added nvflare>=2.5.1 as regular dependency

$ source ~/.local/nvf/.venv/bin/activate
(nvf) ~$
```

if you plan to work with your NVFlare project for a while, you should consider activating it automatically each time you login, e.g. 

```bash
echo 'source ~/.local/nvf/.venv/bin/activate' >> ~/.profile  # or ~/.bash_profile for older Linux
```

Next we will be connecting to an NVFlare Dashboard that someone else set up 

# Connecting to an existing NVFlare Project

In NVFlare, before deploying a new project, you'll likely connect to an existing project that a collaborating institution's `Project Admin` has installed. Your institution will need users with different roles:

- `Lead`: The most common role, typically held by researchers. They submit jobs to the NVFlare system, which then executes the jobs on NVFlare clients (physical or virtual servers) set up by an `Org Admin`.
- `Org Admin`: Can be a researcher with an AWS or Azure cloud account, or someone who can log in as a non-root user to a server with GPUs. They are responsible for installing and managing NVFlare clients. This role can also be held by cloud or research computing administrators. `Org Admins` are not allowed to submit jobs.
- `Member`: Has read-only access to the project by default.

To ensure proper separation of duties, your institution will need at least two accounts: one with the `Org Admin` role for managing infrastructure, and another with the `Lead` role for submitting jobs. Each account requires a different email.

## Using NVFlare as a Lead

`Leads` will sign up and (after approval by the Project Admin) login to the NVFlare dashboard to download authentication and configuration information. We assume the dashboard will be at `https://myproject.mydomain.edu` (your collaborators will share the actual address/URL with you). Once you have registered as a `Lead` and been approved you will be able to login to the dashboard and can download the console. 

![image](https://github.com/dirkpetersen/nvflare-cancer/assets/1427719/fd174c42-c0dc-4fe2-9525-8bfb65529a8a)

To get credentials to the NVFlare system login as `Lead` at `https://myproject.mydomain.edu` and click "Download FLARE Console" under DOWNLOADS and keep the password. The console is downloaded as a zip file called `your@email.adr.zip`. Then unzip the console to a folder in your home directory for this specific NVFlare project and enter the password

```bash
mkdir -p ~/.nvflare/myproject 
unzip -d ~/.nvflare/myproject ./my-lead\@domain.edu.zip
```
then run `~/.nvflare/myproject/my-lead@domain.edu/startup/fl_admin.sh`, enter the email address `my-lead@domain.edu` when prompted and run the command `check_status server` .


### Using the administrative console 

```
~/.nvflare/myproject/my-lead\@domain.edu/startup/fl_admin.sh

> check_status server
Engine status: stopped
---------------------
| JOB_ID | APP NAME |
---------------------
---------------------
Registered clients: 1
----------------------------------------------------------------------------
| CLIENT | TOKEN                                | LAST CONNECT TIME        |
----------------------------------------------------------------------------
| AWS-T4 | 6d49fb6b-32ec-4431-a417-bf023f85f2d0 | Mon May  6 06:28:16 2024 |
----------------------------------------------------------------------------
Done [1087332 usecs] 2024-05-05 23:28:25.033931

```

You are now connected to an NVFlare system. Enter `bye` to leave the console. As a next step let's run a test job using python. 

### Testing a Python example 

We will clone the NVFlare repos into a shared project folder to use some of the standard examples and create a new folder `wksps` and change to it:

```bash
cd /shared/myproject
git clone https://github.com/NVIDIA/NVFlare  # or git@github.com:NVIDIA/NVFlare.git
mkdir NVFlare/wksps
cd NVFlare/wksps
```

Let's copy the standard code sample into the current folder before we run it. 

```bash
cp -rf ../examples/hello-world/hello-numpy-sag ./hello-numpy-sag
```

and then copy the python code below to file `submit-nvflare-job.py` and set project and lead username: 

```python
#! /usr/bin/env python3

import os, sys, argparse
import nvflare.fuel.flare_api.flare_api as nvf

project = "myproject"
username = "my-lead@mydomain.edu"
pocstartup = '/tmp/nvflare/poc/example_project/prod_00'

parser = argparse.ArgumentParser(description='NVFlare job submission with various options')
parser.add_argument('jobfolder', type=str, help='folder that contains an nvflare job, e.g. meta.json')
parser.add_argument('--poc', action='store_true', help='Enable POC mode')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--timeout', type=int, default=3600, help=f'NVFlare time (default: 3600 sec)')
parser.add_argument('--username', type=str, default=username, help=f'Lead user name (default:{username})')
parser.add_argument('--project', type=str, default=project, help=f'A folder under ~/.nvflare (default: {project})')
args = parser.parse_args()

startupkit = os.path.join(os.path.expanduser("~"),
          ".nvflare", args.project, args.username)

if args.poc:
    args.username = 'admin@nvidia.com'
    startupkit = os.path.join(pocstartup,args.username)
    print(f'Launching poc with startup kit: {startupkit}')

sess = nvf.new_secure_session(
    username=args.username,
    startup_kit_location=startupkit,
    debug=args.debug,
    timeout=args.timeout
)

print(sess.get_system_info())

job_folder = os.path.join(os.getcwd(), args.jobfolder) # absolute path required
print(f'Submitting job {job_folder} ...')
job_id = sess.submit_job(job_folder)
print(f"{job_id} was submitted")


sess.monitor_job(job_id, cb=nvf.basic_cb_with_print, cb_run_counter={"count":0})
```

Once you run the job submit script 

```
python3 submit-nvflare-job.py hello-numpy-sag/jobs/hello-numpy-sag
```

you should see something like this 

```
SystemInfo
server_info:
status: stopped, start_time: Sun May  5 23:25:37 2024
client_info:
AWS-T4(last_connect_time: Wed May  8 17:14:01 2024)
job_info:

e8d1e2c9-b47f-43fb-b95a-03551c07b93f was submitted

{'name': 'hello-numpy-sag', 'resource_spec': {}, 'min_clients': 2, 'deploy_map': {'app': ['@ALL']}, 'submitter_name': 'my-lead@domain.edu', 'submitter_org': "FL site", 'submitter_role': 'lead', 'job_folder_name': 'hello-numpy-sag', 'job_id': 'e8d1e2c9-b47f-43fb-b95a-03551c07b93f', 'submit_time': 1715213649.985026, 'submit_time_iso': '2024-05-09T00:14:09.985026+00:00', 'start_time': '', 'duration': 'N/A', 'status': 'SUBMITTED'}
```

### Testing a Pytorch example

The next step is running a Pytorch example with our infratructure. We are going to use the popular [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) example, which includes downloading of some image data. If you have not cloned the NVFlare repository and changed to folder NVFlare/wksps or setup `submit-nvflare-job.py` yet, please see the [Python example](#testing-a-python-example) first.
Then let's copy the hellp-pt example into our working directory (NVFlare/wksps/) and  install the required packages (torch & torchvision)

```bash
cd NVFlare/wksps
cp -rf ../examples/hello-world/hello-pt ./hello-pt
python3 -m pip install -r ./hello-pt/requirements.txt
```

Optional: In a separate terminal you can start a jupyter lab server and continue some of this work from there. The output says somewhere towards the end: **copy and paste one of these URLs** into a browser: 

```
jupyter lab --no-browser --notebook-dir ~/NVFlare/
```

#### Pytorch simulator mode

Next we'll use the nvflare simulator to run a job on 2 clients that is configured inside the workspace: 

```
nvflare simulator -w ./hello-pt -n 2 ./hello-pt/jobs/hello-pt
```

after this you can see the output of this simulation unter `./hello-pt/simulate_jobs`, for example in the `models` sub folder. 

#### Pytorch poc mode

Next we'll run `hello-pt` in [`poc` mode](https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/poc_command.html#poc-command) which a bit closer to production than simulator as it installs all the servers and clients on a single machine, and sets up the same tcp/ip communication path as if they were on different servers. Once `poc` works, you can easily deploy to production. 
For `poc` mode we need to have a single machine with ideally 2 GPUs. If there is an HPC cluster, let's request a machine interactivelty for one day: `srun -p gpu --gres gpu:2 -t 1-0 --pty bash`. Once this is allocated, open up a second terminal window and try to request a second shell on the same machine, assuming that machine is called `node-7`: `srun -w node-7 -p gpu -t 1-0 --pty bash`

Now let's source the virtual envionment again and switch to our workspaces directory in both terminal windows.  

```
source ~/.local/nvf/.venv/bin/activate
cd NVFlare/wksps
```

in the first terminal run these commands: 

```
nvflare poc prepare -n 2
nvflare poc start
```

and in the second terminal you simply run `submit-nvflare-job.py` with the `--poc` option

```
python3 submit-nvflare-job.py --poc hello-pt/jobs/hello-pt
```

and the output looks similar to this:

```
Launching poc with startup kit: /tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com
SystemInfo
server_info:
status: started, start_time: Tue May 28 09:04:52 2024
client_info:
site-1(last_connect_time: Tue May 28 21:14:53 2024)
site-2(last_connect_time: Tue May 28 21:14:49 2024)
job_info:
JobInfo:
  job_id: 972c9a2a-aecd-494c-bc60-6c7e7dc5fc99
  app_name: app
Submitting job /shared/myproject/NVFlare/wksps/hello-pt/jobs/hello-pt/ ...
ea8e0913-fe42-488f-8b37-0c0bcb40a52f was submitted

{'name': 'hello-pt', 'resource_spec': {}, 'min_clients': 2, 'deploy_map': {'app': ['@ALL']}, 'submitter_name': 'admin@nvidia.com', 'submitter_org': 'nvidia', 'submitter_role': 'project_admin', 'job_folder_name': 'hello-pt', 'job_id': 'ea8e0913-fe42-488f-8b37-0c0bcb40a52f', 'submit_time': 1716956099.7929182, 'submit_time_iso': '2024-05-28T21:14:59.792918-07:00', 'start_time': '', 'duration': 'N/A', 'status': 'SUBMITTED'}

```

once poc mode is working you can use `submit-nvflare-job.py` to submit to the production infrastructure, if there is a problem you can use the --debug option to learn more 

```
python3 submit-nvflare-job.py hello-pt/jobs/hello-pt
```

The `submit-nvflare-job.py` script has a number of options

```
python3 submit-nvflare-job.py --help
usage: submit-nvflare-job.py [-h] [--poc] [--debug] [--timeout TIMEOUT] [--username USERNAME] [--project PROJECT] jobfolder

NVFlare job submission with various options

positional arguments:
  jobfolder            folder that contains an nvflare job, e.g. meta.json

options:
  -h, --help           show this help message and exit
  --poc                Enable POC mode
  --debug              Enable debug mode
  --timeout TIMEOUT    NVFlare time (default: 3600 sec)
  --username USERNAME  Lead user name (default:my-lead@mydomain.edu)
  --project PROJECT    A folder under ~/.nvflare (default: myproject)
```

### Client API with external process (Slurm use case)


NVFlare supports the use of [external processes](https://github.com/NVIDIA/NVFlare/blob/main/examples/hello-world/ml-to-fl/README.md#advanced-user-options-client-api-with-different-implementations) which enables us to spawn of a job to an hpc cluster. Please connect to the login node of your HPC cluster and run a POC job after your org admin did [install a client on an HPC login node](#install-a-client-on-the-hpc-login-node):

- You can write your client side training script following examples https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/ml-to-fl

- Then you just use this job template that used FilePipe: https://github.com/NVIDIA/NVFlare/tree/main/job_templates/sag_np and replace executions of `python3` with `srun python3`

- You need to use FilePipe instead of the default CellPipe because CellPipe cannot work in POC mode as it listens on localhost and the Slurm node would not know to which host to talk to.

Let's start this step by step in POC mode. 

- First you connect to the login node and use the [simple instructions for WSL Laptop](#install-a-client-on-your-wsl-laptop) to install the client and switch to the checked out NVFlare git repository and create a `myjobs` folder (assuming you cloned the NVFlare repos in the root of your home directory)

```
cd ~/NVFlare && mkdir myjobs
```

as sag_np template uses FilePipe, I ran this in the cloned NVFlare git repos: 

```
dp@box:~/NVFlare$ nvflare job create -w sag_np -j ./myjobs/slurm_job_fp -sd examples/hello-world/ml-to-fl/np/src/
```

in `./myjobs/slurm_job_fp/app/config/config_fed_client.conf` you replace `app_script = "cifar10.py"` with `app_script = "train_full.py"` and `script = " python3 -u custom....` with `script = "srun python3 -u custom....`

Slurm uses `sbatch` or `srun` to submit jobs. Here you use  `srun python3 -u custom/{app_script} {app_config}` because unlike `sbatch`, `srun` executes synchronous and simply waits until the python script for that specific job has finished.  Of course you can run hundreds of srun sessions in parallel. 

You can also add a few debugging statements to `./myjobs/slurm_job_fp/app/custom/train_full.py` 

When you try POC mode, you need to consider that the POC default workdir is /tmp/nvflare/poc and change it to a workdir in the shared file system (in this case `~/temp/nv-work` in your home directory that is mounted on all compute nodes) so the compute node that is executing train_full.py can also reach the work space. In a separate 2nd terminal on the Slurm login node you run: 

```
export NVFLARE_POC_WORKSPACE=~/temp/nv-work
nvflare poc prepare -n 2
nvflare poc start
```

in a 3rd terminal on the same login node you ask `squeue` to show all your jobs and refresh every second.

```
squeue --me -i 1
```

back to your first terminal you run : 

```
dp@box:~/NVFlare$ nvflare job submit -j ./myjobs/slurm_job_fp
trying to connect to the server
job: '39be4acb-80b3-4e9b-b0e4-31f206578092 was submitted
```

In the squeue terminal you will see 2 jobs are starting and seem to finish normally. Here is the end of the output of the POC process :

```
2024-10-31 21:59:53,862 - ClientRunner - INFO - [identity=site-1, run=39be4acb-80b3-4e9b-b0e4-31f206578092]: Client is stopping ...
2024-10-31 21:59:53,862 - ClientRunner - INFO - [identity=site-2, run=39be4acb-80b3-4e9b-b0e4-31f206578092]: Client is stopping ...
2024-10-31 21:59:55,189 - ReliableMessage - INFO - shutdown reliable message monitor
2024-10-31 21:59:55,190 - ReliableMessage - INFO - shutdown reliable message monitor
2024-10-31 21:59:55,364 - MPM - WARNING - #### MPM: still running thread ClientAPILauncherExecutor_0
2024-10-31 21:59:55,364 - MPM - WARNING - #### MPM: still running thread ClientAPILauncherExecutor_0
2024-10-31 21:59:55,364 - MPM - INFO - MPM: Good Bye!
2024-10-31 21:59:55,364 - MPM - INFO - MPM: Good Bye!
```


#### Connecting to HPC with an older version of NVFlare

If you are using an older version of NVFlare need to ask the Org Admin of your organization [Install a client on an HPC node](#install-a-client-on-hpc). This has the disadvantage that the HPC node needs to be reserved at all times ot be able to accept jobs, blocking GPU hardware even during idle times. However it can be a useful temporary solution as your researchers are transitioning to the lastest version of NVFlare

### Troubleshooting

#### Missing server dependencies

If you get an `Error 113` in the server log, this might mean that a dependency on the server is missing. For example, the NVFlare hello-pt example does not only require Pytorch on the clients but also on the server. To confirm the root cause, use the FLARE console (admin CLI) to login, and execute command download_job [job-id] to get the entire workspace folder. You will find it in the transfer folder of the console. Please check the workspace/log.txt inside the job folder for more details.

#### No default VPC

If you receive a VPC error such as (`VPCIdNotSpecified`) it means that no default network configuration ([Default VPC](https://docs.aws.amazon.com/vpc/latest/userguide/default-vpc.html)) has been created by your AWS administrator. Default VPCs are often used in smaller test envionments. You can create a default VPC by using this command: `aws ec2 create-default-vpc --region us-west-2` . If that fails you may not have permission to create this and have to reach out to your AWS Administrator for a solution. In NVFlare versions >= 2.4.1 you are given an option to pick your own --vpc-id and --subnet-id.

#### SSL issue 

You may get this SSL error in log.txt with some versions of Python and Red Hat linux

```
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)>
```
as a workaround edit file `site-packages/sitecustomize.py`

```
vi $(python -c "import site; print(site.getsitepackages()[0])")/sitecustomize.py
```

and paste in this content, it will fix the issue for all scripts using this Python installation:

```python 
import ssl
try:
    import certifi
    # Define a function that returns a default SSL context with certifi's CA bundle
    def create_certifi_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
        return ssl.create_default_context(purpose, cafile=certifi.where(), capath=capath, cadata=cadata)
    # Set the default SSL context creation function to use certifi's CA bundle
    ssl._create_default_https_context = create_certifi_context
except:
    print('certifi package not installed')
```

## Using NVFlare as an Org Admin

### Register client sites 

If you are the `Org Admin` of a collaborating organization, you join by signing up at `https://myproject.mydomain.edu` with your email address, Name and password. In a second step you are asked to enter your Organization name. Pick `Org Admin` as your role before you add one or more client sites with number of GPUs and memory per GPU. Here you describe the computers with GPUs that you can access to install NVFlare clients. Give them self-explanatory client site names, for example if your site is a single Windows Laptop with an RTX-3080 GPU you may call it WSL-RTX3080. If you have an HPC system with A40 GPUs, call one client site HPC-A40.
For AWS, lets register a client site with a single T4 GPU with 16GB memory, e.g. AWS-T4 (as of May 2024, the lowest cost instance type with a T4 is g4dn.xlarge) 

#### Enter available GPU memory

As you enter the GPU memory for your sites, please note that you need to enter the **actual** GPU memory which is a bit different from what you find in the GPU specs.

| GPU Type | specs (GB) | actual (GB) |
|----------|-------------|-------------|
| A40      | 48          | 44          |
| T4       | 16          | 15          |
| x080ti   | 11          | 10          |
| RTX3080  | 4           | 3           |


### Install a client on AWS 

Login as `Org Admin` at `https://myproject.mydomain.edu` and confirm that you have added a client site, that you perhaps call AWS-T4 based on the GPUs you use. After this client site is approved by the `Project Admin` you can go to DOWNLOADS -> Client Sites -> AWS-T4 click "Download Startup Kit" and keep the password (copy it somewhere)

Move the file to the project folder in your file system, unzip it and paste the password when prompted

```bash
unzip AWS-T4.zip 
cd AWS-T4
```

then you add the packages you need in the client to `startup/requirements.txt` :

```bash
echo -e "torch \ntorchvision \ntensorboard" >> startup/requirements.txt
```

After this, run the `startup/start.sh` script or follow [these instructions to install the client on AWS](https://nvflare.readthedocs.io/en/main/real_world_fl/cloud_deployment.html#deploy-fl-client-on-aws): 

```bash
startup/start.sh --cloud aws     # you can get more automation by using: --config my_config.txt
```

**Note**: If you receive a VPC error such as (`VPCIdNotSpecified`), you may be able to mitigate the issue by using this command: `aws ec2 create-default-vpc --region us-west-2`. You can find more details in the troubleshooting section under [No default VPC](#no-default-vpc)


Now you need to confirm or change a few default settings. After confirming your AWS region you can edit the AMI image name (which supports wildcards *), that is used to search AWS for an AMI image ID for your specific AWS region. Our default here is Ubuntu 24.04 as it has the latest supported Python version (3.12). You can also change amd64 to arm64 as ARM based instances are sometimes lower cost. 
If you are running just a first test, it is fine to not use a GPU machine and instead pick the low cost t2.small instance.

```
Note: run this command first for a different AWS profile:
  export AWS_PROFILE=your-profile-name.

* Cloud EC2 region, press ENTER to accept default: us-west-2
* Cloud AMI image name, press ENTER to accept default (use amd64 or arm64): ubuntu-*-22.04-arm64-pro-server
    retrieving AMI ID for ubuntu-*-24.04-arm64-pro-server ...  ami-04dd23e62ed049936 found
    finding smallest instance type with 1 GPUs and 15360 MiB VRAM ... g5g.xlarge found
* Cloud EC2 type, press ENTER to accept default: g5g.xlarge
* Cloud AMI image id, press ENTER to accept default: ami-04dd23e62ed049936
region = us-west-2, EC2 type = g5g.xlarge, ami image = ami-04dd23e62ed049936 , OK? (Y/n)
If the client requires additional Python packages, please add them to /home/dp/NVFlare/dirk/Test/AWS-T4.X/startup/requirements.txt !
Press ENTER when it's done or no additional dependencies.
```

The output should be similar to this :

```
Checking if default VPC exists
Default VPC found
Generating key pair for VM
Creating VM at region us-west-2, this may take a few minutes ...
VM created with IP address: xx.xxx.xxx.203
Copying files to nvflare_client
Destination folder is ubuntu@xx.xxx.xxx.203:/var/tmp/cloud
Installing os packages with apt in nvflare_client, this may take a few minutes ...
Installing user space packages in nvflare_client, this may take a few minutes ...
System was provisioned
To terminate the EC2 instance, run the following command:
  aws ec2 terminate-instances --region us-west-2 --instance-ids i-0dbbd2fb9a37c6783
Other resources provisioned
security group: nvflare_client_sg_5254
key pair: NVFlareClientKeyPair
review install progress:
  tail -f /tmp/nvflare-aws-YGR.log
login to instance:
  ssh -i /home/dp/NVFlare/NVFlareClientKeyPair_i-0dbbd2fb9a37c6783.pem ubuntu@xx.xxx.xxx.203
```

Now try logging in :

```bash
ssh -i /home/dp/NVFlare/NVFlareClientKeyPair_i-0dbbd2fb9a37c6783.pem ubuntu@xx.xxx.xxx.203
```

or wait until the install has finished, you can check progress in /tmp/nvflare-aws-YGR.log on your machine:

```bash
tail -f /tmp/nvflare-aws-YGR.log
```

In case you forgot to add some packages to requirements.txt, that may be required by other NVFlare examples, you add them an and finally you should delete your pip cache (this can save gigabytes of disk space)

```bash
python3 -m pip install --upgrade --no-cache-dir pandas numpy
rm -rf ~/.cache/pip
```

#### Upgrade all packages

As a last step, make sure the newest packages are installed and that a reboot works

```bash
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y
sudo reboot 
```

### Install a client on HPC 

Note: *This approach requires you to allocate a GPU node for a long time. Running [nvflare on the login node and then submitting jobs to the cluster](#client-api-with-external-process-slurm-use-case) is likelty much more efficient. For this you just need to [Install a client on the HPC login node](#install-a-client-on-the-hpc-login-node)*

Login as `Org Admin` at `https://myproject.mydomain.edu` and confirm that you have added a client site, that you perhaps call HPC-A40 based on the GPUs you use. Again, make sure you enter the [correct amount of GPU memory](#enter-available-gpu-memory). After this client site is approved by the `Project Admin` you can go to DOWNLOADS -> Client Sites -> HPC-A40 click "Download Startup Kit" and keep the password (copy it somewhere)

Move the file to the project folder in your file system, unzip it and paste the password when prompted

```bash
unzip HPC-A40.zip 
cd HPC-A40
```

After changing to the HPC-A40 directory you should see 2 folders and a readme file

```
local
readme.txt
startup
```

Most modern HPC systems will use the Slurm workload manager. On such a system you don't require the overhead of a container or even a virtual machine that needs to be installed first. You can simply submit a batch job that will launch the NVFlare client. In this case we assume that the HPC admin has made A40 GPUs available as general resource (GRES) named gpu:a40 (`--gres gpu:a40`) and we want a single GPU (`--gres gpu:a40:1`). If you don't care about the specific GPU model, you simply request `--gres gpu:1`. (With newer versions of Slurm you can also use `--gpus a40:1` or `--gpus 1`) Let's create a little shell script called `nvflare-HPC-A40.sub` that functions as Slurm submission script:

```bash
#! /bin/bash
#SBATCH --job-name "NVFlare-HPC-A40"
#SBATCH --time 1-00:00:00  # one day
#SBATCH --partition gpu    # if you have your GPUs in an extra queue
#SBATCH --gres gpu:a40:1   # or --gpus a40:1 with recent Slurm versions
#SBATCH --output nvflare-%J.out
#SBATCH --error nvflare-%J.err

nvidia-smi
folder=$(pwd)
client=HPC-A40
organization=Test
source ~/.local/nvf/.venv/bin/activate
python -u -m nvflare.private.fed.app.client.client_train -m ${folder} -s fed_client.json --set uid=${client} secure_train=true config_folder=config org=${organization}
```

As you can see, we run the nvflare.private.fed.app.client.client_train python module with the HPC-A40 configuration in an organization named "Test" 

Now let's run this script in the HPC-A40 folder and then use the `tail -f` command to show the output file in real time. 

```
$ sbatch nvflare-HPC-A40.sub
Submitted batch job 424459530

tail -f nvflare-424459530.out
```

If the output file does not exist, the job has not started yet. In that case run the `squeue --me` command to check the reason why your job may not have started yet. If you find squeue a bit complicated, you can simply use `tsqueue` after installing the [slurm-gui Python Package](https://pypi.org/project/slurm-gui). 

The next issue is that this job will run for a while and then stop when the maximum time is reached. How can you ensure that there is always a client running? As a workaround you can use a helper script that periodically checks if a job with the same job name has been stopped, and launch a new job. For example, you can create a script `nvflare-check-run.sh` that takes the Slurm submission script as an argument:

```bash
#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <slurm_submission_script>"
  exit 1
fi

# Get the submission script from the argument
SUBMISSION_SCRIPT="$1"

# Extract the directory where the submission script resides
SCRIPT_DIR=$(dirname "$SUBMISSION_SCRIPT")

# Change to the script directory
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }

# Get the job name from the submission script (assuming it's defined in the script)
JOB_NAME=$(grep -Eo '^#SBATCH --job-name[= ]+("[^"]+"|\S+)' "$SUBMISSION_SCRIPT" | awk -F'[= ]+' '{print $NF}' | tr -d '"')

if [ -z "$JOB_NAME" ]; then
  echo "--job-name not found in the submission script."
  exit 1
fi

# Check if a job with the same name is currently running
RUNNING_JOB=$(squeue --name="$JOB_NAME" --format="%j" | grep "$JOB_NAME")

if [ -z "$RUNNING_JOB" ]; then
  # No running job found with the same name, submit the new job
  sbatch "$SUBMISSION_SCRIPT"
  echo "Submitted job: $JOB_NAME"
else
  echo "Job $JOB_NAME is still running."
fi
```

and then you simply add this as a cronjob on your HPC login node, that runs perhaps hourly:

```
crontab -e
46 * * * * $HOME/bin/nvflare-check-run.sh /shared/nvf/nvflare-HPC-A40.sub >> /shared/nvf/nvflare-check-run.log 2>&1
```

There are a few considerations when running NVFlare on an HPC Cluster:

- You need to determine how long a job you submitted will be allowed to run. In this example we assume that the job can run for 1 day but the policies at your site may be different
- Check with your HPC admin if it is allowed to run cronjobs on the HPC login node, it will likely be OK if it is a light weight bash script like this one.
- Most HPC nodes need to allocate a GPU exclusively for the duration of the job. We need to understand that NVFlare client will wait for jobs while a GPU is already allocated which means that the GPU will be idle most times
- HPC systems using Slurm >=22.05 have the ability to share GPUs across multiple jobs. You can ask your HPC Admin to enable [Slurm GPU Sharding](https://slurm.schedmd.com/gres.html#Sharding) to increase the efficiency of the HPC cluster.

#### Install a client on the HPC login node 

These instructions are required for uses of [Client API with external process (Slurm use case)](#client-api-with-external-process-slurm-use-case). To run the client on an HPC login node or another node that has access to the srun command we are creating a service. It is recommended to get a service account from the HPC team and to install this in the home director of that service account. First prepare the folder

```
mkdir -p ~/.config/systemd/user
vi ~/.config/systemd/user/nvflare-client.service
```

then create the service file 

```
[Unit]
Description=NVFlare Client Training Service
After=network.target

[Service]
Type=simple
WorkingDirectory=%h
ExecStart=/bin/bash -c 'folder=$(pwd); client=HPC-A40; organization=Test; source %h/.local/nvf/.venv/bin/activate; python -u -m nvflare.private.fed.app.client.client_train -m ${folder} -s fed_client.json --set uid=${client} secure_train=true config_folder=config org=${organization}'
Restart=on-failure
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
```

and finally enable and start the service 

```
systemctl --user daemon-reload
systemctl --user enable nvflare-client.service
systemctl --user start nvflare-client.service
```


### Install a client on your WSL Laptop

To run NVFlare on WSL Linux, you need an NVidia GPU in your Laptop and Windows 11 or a qualifying version of Windows 10. Please [see details here](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).

Login as `Org Admin` at `https://myproject.mydomain.edu` and confirm that you have added a client site, that you perhaps call WSL-RTX3080 based on the GPUs you use. After this client site is approved by the `Project Admin` you can go to DOWNLOADS -> Client Sites -> WSL-RTX3080, click "Download Startup Kit" and keep the password (copy it somewhere)

Move the file to the project folder in your file system, unzip it and paste the password when prompted

```bash
unzip WSL-RTX3080.zip 
cd WSL-RTX3080
startup/start.sh
```

### Verify installation 

Finally we want to check if all components are working together. We will use the [fl_admin.sh script](https://nvflare.readthedocs.io/en/main/real_world_fl/operation.html#operating-nvflare) for this. The script is available in each of the different FLARE Console packages. They contain different credentials for each user that downloads this package. 

To check the server status, login as `Project Admin` or `Org Admin` at https://myproject.mydomain.edu and click "Download FLARE Console" under DOWNLOADS and keep the passsword. 

move the file to the location where you launched the console install earlier, unzip the server startup kit and enter the password

```bash
unzip orgadm\@domain.edu.zip 
cd orgadm\@domain.edu
```

then run startup, enter the email address of the user and run `check_status server` and/or `check_status client`

```
startup/fl_admin.sh

> check_status server
Engine status: stopped
---------------------
| JOB_ID | APP NAME |
---------------------
---------------------
Registered clients: 1
----------------------------------------------------------------------------
| CLIENT | TOKEN                                | LAST CONNECT TIME        |
----------------------------------------------------------------------------
| AWS-T4 | 6d49fb6b-32ec-4431-a417-bf023f85f2d0 | Mon May  6 06:28:16 2024 |
----------------------------------------------------------------------------
Done [1087332 usecs] 2024-05-05 23:28:25.033931

> check_status client
----------------------------------------
| CLIENT | APP_NAME | JOB_ID | STATUS  |
----------------------------------------
| AWS-T4 | ?        | ?      | No Jobs |
----------------------------------------
Done [61900 usecs] 2024-05-05 23:28:34.333857
```

Enter `bye` to leave the console. Once you have installed a new client as an `Org Admin` your users can connect to the system as `Lead`. Please see `Using NVFlare as a Lead`



# (Not needed right now): Deploying a new NVFlare Project

The documentation below is not required for the current project that is exploring NVFlare in cancer research, however it can be useful if you would like to install your own production level NVFlare system

## Installing Dashboard

The NVFlare dashboard will be created in an isolated AWS account. Please see these instructions to [create the dashboard on AWS](https://nvflare.readthedocs.io/en/main/real_world_fl/cloud_deployment.html#create-dashboard-on-aws) or use this command and enter the email address of the `Project Admin` when prompted.

```
nvflare dashboard --cloud aws
```

**Note**: If you receive a VPC error such as (`VPCIdNotSpecified`), you may be able to mitigate the issue by using this command: `aws ec2 create-default-vpc --region us-west-2`. You can find more details in the troubleshooting section under [No default VPC](#no-default-vpc)

After the dashboard is started you will see a dashboard URL that includes an IP address and looks like `http://xxx.xxx.xxx.xxx:443`. Make sure you record the email address and the 5 digit initial password that is displayed in the terminal. Verify that you can login with email address as the user and the password at that URL. You can change your password at `MY INFO -> Edit My Profile`

### Getting dashboard production ready 

Now the dashboard is installed and you would like to use it more permanently, we need to:

1. Ensure that the dashboard will be automatically started after a reboot 
1. Ensure that the dashboard is monitored and restarted on error.
1. Set up DNS/HTTPS to ensure that users don't have to connect to an ip-address insecurely.  

#### About 1. Start at reboot 

Login to the dashboard instance via ssh (using -i NVFlareDashboardKeyPair.pem)  

```bash
ssh -i "NVFlareDashboardKeyPair.pem" ubuntu@52.123.123.123
```

and run this command to add a line to the crontab file:

```bash
(crontab -l 2>/dev/null; echo "@reboot \$HOME/.local/bin/nvflare dashboard --start -f \$HOME > /var/tmp/nvflare-docker-start.log 2>&1") | crontab
```

#### About 2. Restart on Error

As in 1. login with NVFlareDashboardKeyPair.pem and create a script ~/monitor.sh as the Ubuntu user

```bash
#!/bin/bash

url="https://myproject.mydomain.edu"   # Set the website URL
search_string='name="viewport"'        # Set the search string in the HTML source
timeout_duration=15                    # Set the timeout duration in seconds
date=$(date)                           # Get the current date for logging

# Check if the search string exists in the HTML source code
if curl -k -s -m $timeout_duration $url | grep -q "$search_string"; then
    echo "${date}: OK ! The search string '$search_string' was found in the HTML source code of $url"
else
    echo "${date}: Error ! The search string '$search_string' was not found in the HTML source code of $url or the connection timed out after $timeout_duration seconds"
    # Run the commands if the search string is not found or the connection times out
    echo "${date}: Restarting NVFlare dashboard"
    $HOME/.local/bin/nvflare dashboard --stop
    sleep 3
    $HOME/.local/bin/nvflare dashboard --start -f ~
fi
```

and add it as an hourly cronjob :

```bash
(crontab -l 2>/dev/null; echo "59 * * * * \$HOME/monitor.sh >> /var/tmp/nvflare-monitor.log 2>&1") | crontab
```

#### About 3. DNS/HTTPS

Many IT departments advise their users to only log into websites that do offer secure transport (such as https/ssl). To obtain an SSL certificate we need to configure a DNS domain name that is tied to the certificate, but before we can get a DNS entry we need to create floating permanent IP addresses, (In AWS lingo this is an elastic IP address) that will be tied to the machine that runs the dashboard but can also be assinged to another machine later in case of a migration. We will create 2 floating ip addresses, one for the dashboard and the one for the server. Execute this command twice and note each ip address and allocation id.

```bash
aws ec2 allocate-address --domain vpc  # get YOUR_ALLOCATION_ID
```

Then find out the INSTANCE_ID of the instance you created and associate it with the first allocation id.

```bash
aws ec2 describe-instances #  get YOUR_INSTANCE_ID
aws ec2 associate-address --instance-id YOUR_INSTANCE_ID --allocation-id YOUR_ALLOCATION_ID
```

Then there are 2 options: In most cases you will obtain a DNS name and certificate from your IT department. For this example we assume your domain name for the dashboard `myproject.mydomain.edu` and the server is `myproject-server.mydomain.edu`. In some cases someone else may have configured the AWS DNS Service (Route 53) that allows you to manage your own DNS. If you manage your own DNS you can also easily use SSL certs from [Let's Encrypt](https://letsencrypt.org/). Let's work through both options: 

##### DNS and SSL Certificate through your IT department 

Once your DNS entries are setup and you have received your SSL cert for `myproject.mydomain.edu` from IT you can setup secure transport. In most cases you will receive a pem certificate file protected by a password. Upload that file to myproject.mydomain.edu and 

```bash
(nvf) $ scp -i "NVFlareDashboardKeyPair.pem" mycert.pem ubuntu@myproject.mydomain.edu
(nvf) $ ssh -i "NVFlareDashboardKeyPair.pem" ubuntu@myproject.mydomain.edu
```

use the openssl command to extract the x509 certificate and the key file into the ~/cert folder. Restart the container after that. On the myproject.mydomain.edu server paste in these commands:

```bash
openssl rsa -in mycert.pem -out ~/cert/web.key
openssl x509 -in mycert.pem -out ~/cert/web.crt
nvflare dashboard --stop 
nvflare dashboard --start -f ~
```

##### DNS through Route53 and let's encrypt SSL 


On your computer paste in these commands to generate the export statements for AWS credentials  

```bash
aws configure get aws_access_key_id | awk '{print "export AWS_ACCESS_KEY_ID=" $1}'
aws configure get aws_secret_access_key | awk '{print "export AWS_SECRET_ACCESS_KEY=" $1}'
aws configure get aws_session_token | awk '{print "export AWS_SESSION_TOKEN=" $1}'
```

copy these, login to the to the dashboard instance  

```bash
ssh -i "NVFlareDashboardKeyPair.pem" ubuntu@ec2-xxx-xxx-xxx-xxx.us-west-2.compute.amazonaws.com
```

and paste them into the terminal, then install the aws cli and the certbot-dns-route53 package 

```bash
python3 -m pip install awscli certbot-dns-route53
```

To register our IP address in the AWS DNS system called Route 53 we need to first get the ID of the hosted zone (which is a domain or sub domain in DNS, here MYHOSTEDZONEID) 

```bash
aws route53 list-hosted-zones
```

and then register a new hostname in this hosted zone using the aws route53 command. Replace myproject.mydomain.edu with your fully qualified domain name and 123.123.123.123 with the elastic ip address you created earlier. 

```bash
JSON53="{\"Comment\":\"DNS update\",\"Changes\":[{\"Action\":\"UPSERT\",\"ResourceRecordSet\":{\"Name\":\"myproject.mydomain.edu\",\"Type\":\"A\",\"TTL\":300,\"ResourceRecords\":[{\"Value\":\"123.123.123.123\"}]}}]}"
aws route53 change-resource-record-sets --hosted-zone-id MYHOSTEDZONEID --change-batch "${JSON53}"
```

Then we can use run certbot to connect to letsencrypt.org and create our SSL certificate and copy the files to ~/cert

```bash
certbot certonly --dns-route53 --register-unsafely-without-email --agree-tos --config-dir ~/cert --work-dir ~/cert --logs-dir ~/cert -d myproject.mydomain.edu
cp ~/cert/live/myproject.mydomain.edu/fullchain.pem ~/cert/web.crt
cp ~/cert/live/myproject.mydomain.edu/privkey.pem ~/cert/web.key
chmod 600 ~/cert/live/myproject.mydomain.edu/privkey.pem ~/cert/web.key
```

and restart the dashboard server 

```bash
nvflare dashboard --stop 
nvflare dashboard --start -f ~
```

Let's encrypt ssl certs have a reputation of high security because they expire after 90 days but can be renewed automatically. For this we simply setup a cronjob that runs monthly on the first of every month. Note it may not succeed every month as Let's encrypt does not allow renewing certs that are younger than 60 days. Run this command

```bash
(crontab -l 2>/dev/null; echo "0 0 1 * * \$HOME/.local/bin/certbot renew --config-dir \$HOME/cert --work-dir \$HOME/cert --logs-dir \$HOME/cert >> /var/tmp/certbot-renew.log 2>&1") | crontab
```

upgrade the OS and reboot the instance to verify that it will start properly after maintenance 

```bash
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y
sudo reboot 
```

### Configuring Dashboard

Login to the `https://myproject.mydomain.edu` with the email address and 5 digit initial password of the project administrator and make at least the following changes: 

- At MYINFO click "Edit my Profile" and change the password of the project administrator 
- At PROJECT CONFIGURATION enter short name, title and description and click "Publish" to allow users to sign up 
- At SERVER CONFIGURATION enter `myproject-server.mydomain.edu` at "Server (DNS Name)"
- At PROJECT HOME click "Freeze Project"

After you have frozen the project you can no longer change any of the settings above, however you can still reset the entire dashboard if needed. 

### Resetting Dashboard

If you made a mistake when setting up the dashboard, for example, you froze the project but want to make changes, you can reset the dashboard by moving the datadase out of the way, without having to reinstall the dashboard server.

```bash
nvflare dashboard --stop
mv ~/db.sqlite ~/old.db.sqlite 
nvflare dashboard --start -f $HOME --cred project_adm\@mydomain.edu:mysecretpassword
```

## Installing Server

The server is a central system that is installed alongside the dashboard by the project administrator. The users from the federated sites will connect first to the dashbard to download a configuration and then install a client to build a connection to the server to exhange ML models and meta information but not share their data.

As a project admin login to the dashboard and download the server startup kit and copy the password 

![image](https://github.com/dirkpetersen/nvflare-cancer/assets/1427719/523aa5a0-7c28-4713-b629-352caa90ea4c)

move the file to the location where you launched the console install earlier, unzip the server startup kit and enter the password

```bash
unzip myproject-server.mydomain.edu.zip 
cd myproject-server.mydomain.edu 
```

follow [these instructions to install the server on AWS](https://nvflare.readthedocs.io/en/main/real_world_fl/cloud_deployment.html#deploy-fl-server-on-aws) or execute this command: 

```bash
startup/start.sh --cloud aws     # this is only needed for full automation: --config my_config.txt
```

then we are prompted, and instead the default AMI (Ubuntu 20.04) we pick the newer 24.04 ( ami-04dd23e62ed049936)

```
Cloud AMI image, press ENTER to accept default ami-04bad3c587fe60d89:  ami-04dd23e62ed049936
Cloud EC2 type, press ENTER to accept default t2.small:
Cloud EC2 region, press ENTER to accept default us-west-2:
region = us-west-2, ami image = ami-04dd23e62ed049936, EC2 type = t2.small, OK? (Y/n) Y
If the server requires additional dependencies, please copy the requirements.txt to myproject-server.mydomain.edu/startup/.
Press ENTER when it's done or no additional dependencies.
```

Note: Do NOT put a `requirements.txt` file into the startup folder at this time. You need to increase the size of the file system of this server, before you can install other packages such as pytorch as they are large and come with many dependencies. 

After this you should get confirmation that `myproject-server.mydomain.edu` was installed. You should now [increase the disk (EBS volume)](https://docs.aws.amazon.com/ebs/latest/userguide/requesting-ebs-volume-modifications.html) to at least 16GB 

Then find out the INSTANCE_ID of the instance you created and associate it with the second allocation id for the elastic IP you created earlier: 

```bash
aws ec2 describe-instances #  get YOUR_INSTANCE_ID
aws ec2 associate-address --instance-id YOUR_INSTANCE_ID --allocation-id YOUR_ALLOCATION_ID
```

test your connection with 

```bash
ssh -i NVFlareserverKeyPair.pem ubuntu@myproject-server.mydomain.edu
```


Then make sure you install some packages that may be required by NVFlare examples such as 'hello-pt' and finally delete your pip cache (this can save gigabytes of disk space)

```
python3 -m pip install --upgrade --no-cache-dir torch pandas
rm -rf ~/.cache/pip
```

If there is not enough disk space, you should [increase the partition and file system size](https://docs.aws.amazon.com/ebs/latest/userguide/recognize-expanded-volume-linux.html) of your instance to the disk size you set before (using growpart and resizefs). After that you can try installing python packages again

```
sudo growpart /dev/xvda 1
sudo resize2fs /dev/xvda1
```


Finally make sure the newest packages are installed and that a reboot works

```bash
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y
sudo reboot 
```

## Installing Client

please see [Using NVFlare as an Org Admin](#using-nvflare-as-an-org-admin)

# Contributing code to NVFlare 

If you would like to make a code contribution to NVFlare, please check the [contributor docs](https://nvflare.readthedocs.io/en/main/contributing.html) first. 
In our case, we have made modifications to the cloud deployment scripts and constributed some changes back. Please take these steps after [creating a Fork](https://github.com/NVIDIA/NVFlare/fork) of NVFlare: 

```
git clone git@github.com:your-organization/NVFlare.git
git clone git@github.com:dirkpetersen/nvflare-cancer.git
cd NVFlare
```

check folder `nvflare\lighter\impl` and make modifications to `aws_template.yml` and/or  `master_template.yml`. Then generate a new aws_start.sh script for an NVFlare client in one of your client starter kits startup folder: 

```
../nvflare-cancer/make-aws-client-script.py /starter-kit-folder/startup/aws_start.sh
```

Test this aws_start.sh script intensely before you run `runtest.sh` and commit the code to your forked NVFlare repository and then create a pull request in Github. The make-aws-client-script.py uses the NVFlare internal machinery to generate shell scripts from yaml files:

```python
#! /usr/bin/env python3

import os, sys
from nvflare.lighter import tplt_utils, utils

client = "AWS-T4"
org = "Test"

lighter_folder = os.path.dirname(utils.__file__)
template = utils.load_yaml(os.path.join(lighter_folder, "impl", "master_template.yml"))
template.update(utils.load_yaml(os.path.join(lighter_folder, "impl", "aws_template.yml")))
tplt = tplt_utils.Template(template)
csp = 'aws'
if len(sys.argv) > 1:
    dest = sys.argv[1]
else:
    dest = os.path.join(os.getcwd(), f"{csp}_start.sh")
script = template[f"cloud_script_header"] + template[f"{csp}_start_sh"]
script = utils.sh_replace(
            script, {"type": "client", "inbound_rule": "", "cln_uid": f"uid={client}", "ORG": org}
        )
utils._write(
    dest,
    script,
     "t",
     exe=True,
)
print(f"Script written to {dest} !")
```


