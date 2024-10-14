# Summary Tool Readme

This readme file provides instructions for setting up and using the Summarization Tool, which is built using Flask and leverages the Hugging Face Transformers library to perform question and answer tasks. Before running the tool, you need to ensure that you have Flask installed and the necessary model loaded from a provided link.

## Prerequisites

Before using the Summarization Tool, make sure you have the following prerequisites installed on your system:

- Python 3.x
- Flask
- Hugging Face Transformers library

You can install Flask and the Transformers library using pip:

```bash
pip install flask transformers
```

# Getting Started
- Clone or download the Summarization Tool repository to your local machine.
- Navigate to the project directory:

```
cd summarization-tool
```
Run the Flask application:
```
python app.py
```
This will start the Flask development server, and the Summarization Tool will be accessible at http://localhost:5000 in your web browser.

# Model Loading
Before using the Summarization Tool, you need to load the model from the provided link in the notebook folder:

Model link: https://colab.research.google.com/drive/1dx1hDY7Cny8y7Gak3USn6MTxlu5YQ13N?usp=sharing
Please download the model and place it in a directory within the project folder.

# Question and Answer Generation
Download the model from the hugging face [https://huggingface.co/valhalla/t5-small-qa-qg-hl].


# Making Changes to pipelines.py
To customize or modify the behavior of the model within the Summarization Tool, you may need to make changes to the pipelines.py file. This file contains the code for loading and interacting with the model.

If you need to make changes to pipelines.py to suit your project's requirements, please reach out to the developer for guidance and assistance. You can contact the developer for assistance by using the contact information provided in the project documentation.


**Thank you for using the Summarization Tool!**
