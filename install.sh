# Go to TextMiningCourse directory
python3 -m venv tm_env
source tm_env/bin/activate
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
