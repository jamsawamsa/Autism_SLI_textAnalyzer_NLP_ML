# Autism_SLI_textAnalyzer_NLP_ML
Utilizing Natural Language Processing and Machine Learning, we analyze text transcripts of children (made available via the CHILDES database: http://childes.talkbank.org/) and clinically access whether they have Autistic Spectrum Disorder, Specific Language Impairment or have Typical Development. This was written for my Final Year Project in university. Written in Python3.

Read the full report of the project here: https://github.com/jamsawamsa/Autism_SLI_textAnalyzer_NLP_ML/blob/master/Use_of_NLP_and_ML_to_clinically_access_children.pdf

## Dependencies
For this project, I used 4 libraries for python:
 - [nltk](http://www.nltk.org/)
 - [spaCy](https://spacy.io/)
 - [scikit-learn](http://scikit-learn.org/)
 - [matplotlib](https://matplotlib.org/)
 
 ## Setup and Run
 Install the dependencies. Make sure to download these files and put them in your working directory, these files contain the source code for the program:
  - LanguageModel.py
  - classifier_m.py
  - features.py
  - driver_m.py
  - subject_object_extraction.py
  
 Download the two .7z files and extract them into your working directory:
  - plain_text.7z
  - xml.7z
  
These are the database files, obtained from the CHILDES database. Ensure that your overall folder structure is like this:
  ```
project
│   README.md
│   source code.py files
│
└───xml
│   └───asd
│       │   xmlfile1.xml
│       │   xmlfile2.xml
│       │   ...
│   └───sli
│       │   xmlfile1.xml
│       │   xmlfile2.xml
│       │   ...
│   └───typ
│       │   xmlfile1.xml
│       │   xmlfile2.xml
│       │   ...
└───plain_text
│   └───asd
│       │   chafile1.cha
│       │   chafile2.cha
│       │   ...
│   └───sli
│       │   chafile1.cha
│       │   chafile2.cha
│       │   ...
│   └───typ
│       │   chafile1.cha
│       │   chafile2.cha
│       │   ...
│   
```
You'll also have to modify lines 19 and 20 in the driver_m.py file to point to your directories. 
![alt text](https://github.com/jamsawamsa/Autism_SLI_textAnalyzer_NLP_ML/blob/master/Images/ReadmeImage1.JPG?raw=true)

You can add additional databases from CHILDES if you want, but you'll have to make sure that they have the correct annotations for the program to work. A lot of databases don't have the same annotation and recording protocols so this requires a little bit of research on your end. Ultimately your goal is to have **both** the xml and plaintext versions of the corpora downloaded and sorted according to their labels.

## Workflow
The general project workflow is like so:
1. Organize your folder structure.
2. Run driver_m to generate the "output_file" file.
3. Run classifier_m, which uses the previously generated output_file. classifier_m generates "report.txt" which contains the results of the analysis.


I've included both a sample output_file and results.txt file in the repo. The output_file stores the feature sets of the data and the results.txt is a the report of the analysis, with accuracy and all kinds of useful information. Do check it out.

## Future Work
I admit it's a little cumbersome to use right but this was done in true assingment fashion so I had a lot of time constraints and never really revisited it. I can think of a few ways to improve what's here now and I'll probably get on it if there are enough people using this. A few things I can think of right now are:
 - An extension for this program to accept additional text corpora from CHILDES
 - A corpus pre-processor/cleaner to prepare various kinds of text data
 - An option to choose your path to the folders
 
 ## Authors
 [Jamsawamsa's Github](https://github.com/jamsawamsa)


