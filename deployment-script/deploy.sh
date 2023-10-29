#!/bin/bash

instll_docker() {
    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    # Add the repository to Apt sources:
    echo \
        "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

update_permission() {
    local groupname=$1
    sudo usermod -aG $groupname $USER
}

install_cuda_driver(){
    sudo apt-get install linux-headers-$(uname -r)
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
    wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-drivers
}

install_nvidia_docker(){
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit-base
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
        && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
}

start_docker_torch(){
    sudo docker run --name nvidia_pytorch -itd --runtime=nvidia --gpus all -v /home/$USER/workspace:/workspace nvcr.io/nvidia/pytorch:22.08-py3
}

update_torch_in_docker(){
    if [ "$1" = "falcon-40b-instruct-8bit-pytorch" ]; then
        dependencies="torch torchvision torchaudio transformers accelerate bitsandbytes einops"
    elif [ "$1" = "openthaigpt-1.0.0-beta-8bit-pytorch" ] || [ "$1" = "openthaigpt-1.0.0-beta-pytorch" ] ; then
        dependencies="torch torchvision torchaudio transformers[sentencepiece] accelerate bitsandbytes einops git+https://github.com/huggingface/peft.git"
    else
         exit 0
    fi
    docker exec nvidia_pytorch sh -c "#!/bin/bash
pip3 uninstall --yes torch 
pip3 install $dependencies
"
}

clone_project(){
    local cloud=$1
    local example=$2
    docker exec nvidia_pytorch sh -c "#!/bin/bash
git clone https://github.com/vultureprime/deploy-ai-model.git && cd ./deploy-ai-model/$cloud-example/$example
mkdir llama2-openthaigpt && python3 app.py
"
}

check_nvidia_docker_configure(){
    nvidia_path=$(sudo jq -r '.runtimes.nvidia.path' /etc/docker/daemon.json)
    echo "$nvidia_path"
    if [ "$nvidia_path" = "nvidia-container-runtime" ]; then
        return 1
    else
        return 0
    fi
}

check_gpu_a10g(){
    gpu_list=$(sudo lshw -C display | grep -o 'product: .*' | awk '{print $2$3}')
    if [[ $gpu_list == *'GA102GL[A10G]'* ]]; then
        echo "The gpu contains 'GA102GL[A10G]'."
        return 1
    else
        echo "The gpu does not contain 'GA102GL[A10G]'."
        return 1
    fi
}

main_init(){
    sudo apt install jq
    echo "####### Check GPU requirement #######"
    check_gpu_a10g
    if command -v docker &>/dev/null; then
        echo "Docker is installed on this system."
        update_permission "docker"
    else
        instll_docker
        update_permission "docker"
    fi
    if command -v nvidia-smi &>/dev/null; then
        echo "Nvidia-driver is installed on this system."
    else
        echo "Nvidia-driver is not installed on this system."
        echo "Install Nvidia-driver"
        install_cuda_driver
    fi

    check_nvidia_docker_configure
    result=$?

    if [ $result -eq 0 ]; then
        echo "Configure Nvidia Docker"
        install_nvidia_docker
    else
        echo "Nvidia-driver is installed on this system."
    fi
}

beam_initial_configuration(){
    curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh
    sudo apt install python3-pip
    pip install beam-sdk
    read -p "Beam clientId : " clientId
    read -p "Beam clientSecret : " clientSecret
    beam configure --clientId=$clientId --clientSecret=$clientSecret
}

beam_clone_project(){
    local cloud=$1
    local project=$2
    git clone https://github.com/vultureprime/awesome-beam-cloud.git /home/$USER/$cloud
    cd /home/$USER/$cloud/beam-cloud-exmaple/$project
    pwd
}




beam_example(){
    local beam_project_list=("openthaigpt-1.0.0-beta-pytorch" "Exit")
    echo "Select an example:"
    iter=0
    for item in "${beam_project_list[@]}"; do
        iter=$((iter+1))
        echo "$iter. $item"
    done
    # Read user's choice
    read -p "Enter your choice (1-${#beam_project_list[@]}): " choice
    # Use the case statement to process the choice
    case $choice in
        1)
            project=${beam_project_list[$((1-$choice))]}
            echo "#### You chose $project ####"
            beam_clone_project $cloud $project
            ;;

        2)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please select a valid option."
            ;;
    esac
}

paperspace_example(){
    check_gpu_a10g
    local paperspace_example_list=("falcon-40b-instruct-8bit-pytorch" "openai-langchain-basic-RAG" "openthaigpt-1.0.0-beta-8bit-pytorch" "openthaigpt-1.0.0-beta-pytorch" "Exit")
    echo "Select an example:"
    iter=0
    for item in "${paperspace_example_list[@]}"; do
        iter=$((iter+1))
        echo "$iter. $item"
    done
    # Read user's choice
    read -p "Enter your choice (1-${#paperspace_example_list[@]}): " choice
    # Use the case statement to process the choice
    case $choice in
        1)
            example=${paperspace_example_list[$((1-$choice))]}
            echo "#### You chose $example ####"
            start_docker_torch
            update_torch_in_docker $example
            clone_project $cloud $example
            ;;
        2)
            example=${paperspace_example_list[$((1-$choice))]}
            echo "#### You chose $example ####"
            ;;
        3)
            example=${paperspace_example_list[$((1-$choice))]}
            echo "#### You chose $example ####"
            start_docker_torch
            update_torch_in_docker $example
            clone_project $cloud $example
            ;;
        4)
            example=${paperspace_example_list[$((1-$choice))]}
            echo "#### You chose $example ####"
            start_docker_torch
            update_torch_in_docker $example
            clone_project $cloud $example
            ;;
        *)
            echo "Invalid choice. Please select a valid option."
            ;;
    esac
}


# Display a menu
echo "Select a cloud:"
echo "1. AWS"
echo "2. Beam Cloud"
echo "3. Paperspace"
echo "4. Exit"

# Read user's choice
read -p "Enter your choice (1-4): " choice
# Use the case statement to process the choice
case $choice in
    1)
        echo "You chose AWS"
        cloud="aws"
        ;;
    2)
        echo "You chose Beam"
        cloud="beam"
        beam_initial_configuration
        beam_example
        ;;
    3)
        echo "You chose paperspace"
        cloud="paperspace"
        main_init
        paperspace_example
        ;;
    4)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please select a valid option."
        ;;
esac