# Define variables for Python and pip
PYTHON = python
PIP = pip

# Target to install dependencies
install:
	$(PIP) install -r requirements.txt

# Target to preprocess the data
preprocess:
	$(PYTHON) preprocess.py

# Target to train the model
train:
	$(PYTHON) train_model.py

# Target to generate recommendations for a specific user
recommend:
	$(PYTHON) recommend.py

visualize:
	$(PYTHON) visualize.py

# Target to run all unit tests
test:
	$(PYTHON) -m pytest tests/

# Target to clean up temporary or output files
clean:
	rm -rf __pycache__/ outputs/*.png recommendations/*.csv
