# Go to TextMiningCourse directory
python3 -m venv tm_env
source tm_env/bin/activate
pip3 uninstall neuralcoref
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip3 install -r requirements.txt
pip3 install -e .

# Go back to the root folder
cd ..

pip3 uninstall spacy
pip3 install spacy==2.2.4
python3 -m spacy download en_core_web_sm

# Other dependencies

pip3 install -r extra_requirements.txt

# Run the pd_pedia docker
# docker run -itd --restart unless-stopped -p 2222:80 dbpedia/spotlight-english spotlight.sh

# Make sure to set the environment to the tm_env environment
