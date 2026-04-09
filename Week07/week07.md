# Task 1: Automatic annotation of sample texts with the USAS online tagger
• Create a small corpus of sample texts e.g. news articles downloaded from today’s BBC News (https://www.bbc.co.uk/news) or a similar source
• Copy and paste the contents of each text one by one into the online USAS tagger (https://ucrel-api.lancaster.ac.uk/usas/tagger.html) in order to semantically tag the texts
• View the tagged output and the tag descriptions on the following pages and see if you can understand why certain words and MWEs have been tagged with these USAS labels, and can you spot places where the tags are incorrect?

• https://ucrel.lancs.ac.uk/usas/

• https://ucrel.lancs.ac.uk/usas/USASSemanticTagset.pdf

# Task 2 Install and run the PyMUSAS semantic tagger
• Either, follow the code templates in the how-to guides (https://ucrel.github.io/pymusas/) to create your own Python code to semantically tag your corpus of BBC News texts
• Or, use the existing Python Notebook and run this using Google Colab to semantically tag your texts and compare them:
• https://github.com/UCREL/pymusas_notebook/blob/main/PyMUSAS.ipynb
• Pick a new text in a language that you don’t understand e.g. Spanish and use the semantic tagger for that language to see if you can understand the main key topics in the text without using Google Translate

# Task 3 Spatial Humanities applications
• In the Spatial Humanities project (https://spacetimenarratives.github.io/), we are using the semantic tagger to pick out key concepts to do with emotion (E), Movement, travel and transport (M), and time (T) to help with our research questions on extracting spatial entities from text
• There are multiple Python notebooks and a pip installable library which you can test out to see how this works:
• https://github.com/SpaceTimeNarratives/demo
• https://github.com/SpaceTimeNarratives/spatio-textual