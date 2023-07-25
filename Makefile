ENV_NAME := pgl
PYTHON_VERSION := 3.8

# create_envターゲット: condaで新しい環境を作成して依存関係をインストール
create_env:
	@echo "Creating environment..."
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) -y
	@echo "Installing python dependencies..."
	conda run -n $(ENV_NAME) conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
	conda run -n $(ENV_NAME) pip install -r requirements.txt
	cp -r /nas/data/syamagami/GDA/data/GDA_DA_methods/data ./
	mv ./data/Office31 ./data/office
	mv ./data/OfficeHome ./data/home

# remove_envターゲット: condaの環境を削除
remove_env:
	@echo "Removing environment..."
	conda env remove --name $(ENV_NAME)
	@echo "Environment removed successfully."