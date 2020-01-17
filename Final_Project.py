from bs4 import BeautifulSoup
import requests
import pandas as pd
from pandas.io.html import read_html
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import numpy as np

import textract
import PyPDF2 as p2
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from collections import Counter
import string

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib.colors import LinearSegmentedColormap

import glob	#helps us store images into a specific list and load files within folder
from PIL import Image, ImageDraw


#--------------BLOCK 1 ------------------

def webscraping_csv_visualization(f):

	url="https://en.wikipedia.org/wiki/List_of_ancient_Greek_philosophers"
	page=requests.get(url)	#define a variable to request the page
	soup=BeautifulSoup(page.text,'html.parser')	#parse it according to html format
	table=soup.find('table', {'class':'wikitable sortable'}).tbody
	#if we inspect the html script, we see the data we want to extract under this class
	rows=table.find_all('tr') #tr=all data of the table
	columns=[v.text.replace('\n','') for v in rows[0].find_all('th')] # th= the names of the headers (name,school etc.).  I remove new lines in all headers

	df=pd.DataFrame(columns=columns) #I specify the columns in the dataframe and start creating my CSV file
	for i in range(1,len(rows)):
		tds=rows[i].find_all('td')
		if len(tds)==4: #I need four columns 
			values=[tds[0].text.replace('\n',''), tds[1].text.replace('\n',''),tds[2].text.replace('\n',''),tds[3].text.replace('\n','')] #remove new lines in the rows
		df=df.append(pd.Series(values, index=columns),ignore_index=True)
		#print(df)
		df.to_csv('Philosophers.csv')


		#From this csv I will create a new one to keep only the columns I need for the visualization and specific data as well

	new=df.loc[(df["School"] =='Pythagorean' ) | (df["School"] =='Presocratic, Milesian')| (df["School"]=='Milesian') | (df["School"]=='Ephesian')|(df["School"]=='Eleatic') | (df["School"]=='Pluralist') | (df["School"]=='Atomist')| (df["School"]=='Sophists') | (df["School"] =='Presocratic, Ephesian') |(df["School"] =='Presocratic, Eleatic') | (df["School"] =='Presocratic, Pluralist') | (df["School"] =='Presocratic, Atomist')].drop(columns="Notes").replace('Presocratic, Milesian','Milesian').replace('Presocratic, Ephesian','Ephesian').replace('Presocratic, Pluralist','Pluralist').replace('Presocratic, Eleatic','Eleatic').replace('Presocratic, Atomist','Atomist')



	new.to_csv('Pre-Socratics.csv',index=False)# index false means that I do not keep the indexes(numbers) in the first column, which doesn't exist in the original csv


	


	



#--------------BLOCK 2 ------------------

Thales=[]
Democritus=[]
Heraclitus=[]
Anaxagoras=[]
Parmenides=[]
Pythagoras=[]
remove_stops=[]

def fragments(f):
	with open('Thales.pdf','rb') as a, open ('Democritus.pdf','rb') as b, open ('Heraclitus.pdf','rb') as c, open ('Anaxagoras.pdf','rb') as d, open ('Parmenides.pdf','rb') as e,  open('Pythagoras.pdf','rb') as f,open ('my_stop_words.txt','r') as g:

		#open pdfs in binary mode and also a txt I created with specific stop words

		g_contents=g.read().split('\n') #read stopwords from file
		

		pdf=p2.PdfFileReader(a)	
		i=0
		while i<pdf.getNumPages(): #create a loop to access all pdf's pages
			pageinfo=pdf.getPage(i)#access first page of pdf
			all_text=pageinfo.extractText()	#extract text from pages
			i=i+1 #access the rest of the pages
			Thales.append(all_text) #to exit the loop, I add all my pdf text in a list so that I can work with it later on
		listToStr = ' '.join([str(elem) for elem in Thales]) #convert my list to string
		tokens = nltk.word_tokenize(listToStr)
		remove_numbers=[''.join(x for x in i if x.isalpha()) for i in tokens]#remove numbers from pdf
		words=[letter.lower() for letter in remove_numbers] #lower all letters to remove
		remove_stops1=[word for word in words if word not in stopwords.words('english') + list(string.punctuation)+list(g_contents)] #remove punctuation, stopwords from nltk package and from my txt 
		most_frequent_words= Counter(remove_stops1).most_common(30)
		
		print (most_frequent_words)
		print ('')
		

		pdf=p2.PdfFileReader(b)
		i=1
		while i<pdf.getNumPages():
			pageinfo=pdf.getPage(i)
			all_text=pageinfo.extractText()
			i=i+1
			Democritus.append(all_text)
		listToStr = ' '.join([str(elem) for elem in Democritus]) 
		tokens = nltk.word_tokenize(listToStr)
		remove_numbers=[''.join(x for x in i if x.isalpha()) for i in tokens]
		words=[letter.lower() for letter in remove_numbers]
		remove_stops2=[word for word in words if word not in stopwords.words('english')+ list(string.punctuation)+list(g_contents)]	
		most_frequent_words2= Counter(remove_stops2).most_common(30)
		
		print (most_frequent_words2)
		print ('')
		
		
		pdf=p2.PdfFileReader(c)
		i=0
		while i<pdf.getNumPages():
			pageinfo=pdf.getPage(i)
			all_text=pageinfo.extractText()
			i=i+1
			Heraclitus.append(all_text)
		listToStr = ' '.join([str(elem) for elem in Heraclitus]) 
		tokens = nltk.word_tokenize(listToStr)
		remove_numbers=[''.join(x for x in i if x.isalpha()) for i in tokens]
		words=[letter.lower() for letter in remove_numbers]
		remove_stops3=[word for word in words if word not in stopwords.words('english')+ list(string.punctuation)+list(g_contents)]	
		most_frequent_words3 = Counter(remove_stops3).most_common(30)
		
		print(most_frequent_words3)
		print('') 


		pdf=p2.PdfFileReader(d)
		i=0
		while i<pdf.getNumPages():
			pageinfo=pdf.getPage(i)
			all_text=pageinfo.extractText()
			i=i+1
			Anaxagoras.append(all_text)
		listToStr = ' '.join([str(elem) for elem in Anaxagoras]) 
		tokens = nltk.word_tokenize(listToStr)
		remove_numbers=[''.join(x for x in i if x.isalpha()) for i in tokens]
		words=[letter.lower() for letter in remove_numbers]
		remove_stops4=[word for word in words if word not in stopwords.words('english')+ list(string.punctuation)+list(g_contents)]	
		most_frequent_words4 = Counter(remove_stops4).most_common(30)
		
		print(most_frequent_words4)
		print('') 


		pdf=p2.PdfFileReader(e)
		i=0
		while i<pdf.getNumPages():
			pageinfo=pdf.getPage(i)
			all_text=pageinfo.extractText()
			i=i+1
			Parmenides.append(all_text)
		listToStr = ' '.join([str(elem) for elem in Parmenides])
		tokens = nltk.word_tokenize(listToStr)
		remove_numbers=[''.join(x for x in i if x.isalpha()) for i in tokens]
		words=[letter.lower() for letter in remove_numbers]
		remove_stops5=[word for word in words if word not in stopwords.words('english')+ list(string.punctuation)+list(g_contents)]	
		most_frequent_words5 = Counter(remove_stops5).most_common(30)
		
		print(most_frequent_words5)
		print('') 


		pdf=p2.PdfFileReader(f)
		i=0
		while i<pdf.getNumPages():
			pageinfo=pdf.getPage(i)
			all_text=pageinfo.extractText()
			i=i+1
			Pythagoras.append(all_text)
		listToStr = ' '.join([str(elem) for elem in Pythagoras]) 
		tokens = nltk.word_tokenize(listToStr)
		remove_numbers=[''.join(x for x in i if x.isalpha()) for i in tokens]
		words=[letter.lower() for letter in remove_numbers]
		remove_stops6=[word for word in words if word not in stopwords.words('english')+ list(string.punctuation)+list(g_contents)]	
		most_frequent_words6 = Counter(remove_stops6).most_common(30)
		
		print(most_frequent_words6)
		print('') 

		remove_stops=remove_stops1+remove_stops2+remove_stops3+remove_stops4+remove_stops5+remove_stops6
		#create a list with text from all pdfs without stop words to use later on
		
		
	with open ('frequent_words.txt','w') as output:
		for row in remove_stops:
			output.write(str(row) + '\n')

			#list to txt file



#--------------BLOCK 3 ------------------


def wordcloud(f):
	#CREATE A WORDCLOUD

	matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)

	mask=np.array(Image.open("Thales.jpg")) #use this picture for mask

	#READING THE SCRIPT
	text=open("frequent_words.txt").read()


	#GENERATE WordCloud
	wc=WordCloud(background_color='white',  width=800,height=500,mask=mask, colormap=matplotlib.cm.inferno)
	#Show the WodCloud
	wc.generate(text)

	plt.figure()
	plt.imshow(wc,interpolation='bilinear')
	plt.axis('off') # axis doesn't appear to my png
	plt.savefig('data.png')
	



#--------------BLOCK 4 ------------------


Philosophers=[]
Philosophers2=[]
FinalCollage=[]

def open_analyze_img(f): 

	#COLLAGE 1

	for filename in glob.glob('Philosophers/*.jpg'): #open all the images in this folder
		im=Image.open(filename)
		im_resized=im.resize((128,115))#resize images to fit in the collage, leave some empty space in the bottom
		Philosophers.append(im_resized) #append resized images in list

	size=128	#standard for squared shape, pixels for a single image
	new_im=Image.new('RGB', (384,128)) # a 3x1 collage (3 pictures in one row)
	x=0
	y=0
	for im in Philosophers: #for all images in the list we create thumbnails
		if (x==384):	#if x is incremented to go out of the binaries of the size of the image,
			y=y+size	#y should be incremented (move a column down)
			x=0			# and x should be reset to 0
		im.thumbnail((size,size))	#resize the images and specify the size
		new_im.paste(im,(x,y))		#paste the image into a new image to the position x,y
		x=x+size					#when the next iterration comes in,  x needs to be incremented
	new_im.save('collage.png')



	image=Image.open("collage.png") #open the collage to add text underneath
	d = ImageDraw.Draw(image)
	d.text((40,115), "Democritus", fill=(255,255,0)) #set position and color of the text
	d.text((160,115), "Pythagoras", fill=(255,255,0))
	d.text((290,115), "Anaxagoras", fill=(255,255,0))
	image.save('collage.png')


	#COLLAGE 2

	for filename2 in glob.glob('Philosophers2/*.jpg'):
		im2=Image.open(filename2)
		im2_resized=im2.resize((128,115))
		Philosophers2.append(im2_resized)


	size2=128	
	new_im2=Image.new('RGB', (384,128)) 
	x=0
	y=0
	for im in Philosophers2: 
		if (x==384):	
			y=y+size	
			x=0			
		im2.thumbnail((size2,size2))	
		new_im2.paste(im,(x,y))		
		x=x+size					
	

	new_im2.save('collage2.png')

	image2=Image.open("collage2.png")
	d = ImageDraw.Draw(image2)
	d.text((40,115), "Thales", fill=(255,255,0)) 
	d.text((160,115), "Parmenides", fill=(255,255,0))
	d.text((290,115), "Heraclitus", fill=(255,255,0))
	image2.save('collage2.png')


	#FINAL IMAGE
	
	im1 = Image.open('collage.png')
	im2 = Image.open('collage2.png')

	dst = Image.new('RGB', (im1.width, im1.height + im2.height))
	dst.paste(im1,(0,0))
	dst.paste(im2,(0,im1.height)) #add the second image under the first
	dst.save('final_collage.png')

#since my two final images have the same dimentions, all I have to do is simply cncatenate them

def visualization(f):
	external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
	app = dash.Dash(__name__, external_stylesheets=external_stylesheets) #define the app
	dfWords = pd.read_csv('Pre-Socratics.csv') #set variable for reading the file
		


	app.layout= html.Div(children=[ #to have more than one visualizations
			html.H1(children='Final Project',style={"textAlign": "center"}),#specify the header/title position
			html.Div(children='''Block 1'''),#specify a new place holder

			dcc.Graph(
		        id='scatter-chart',
		        figure={
		            'data':[
		                go.Scatter(
		                    x=list((dfWords.iloc[:,0])), #select  specific columns for x and y
		                    y=list((dfWords.iloc[:,2])),
		                    mode='markers',#"bullets" on scatter plot
		                    opacity=1.0,
		                    marker={	
		                        'size': 10,
		                        'line':{'width':0.9,'color':'red'},#color around marker
		                        },
		                        ),
		                ],
		            'layout': go.Layout(
		                {'title': 'School distribution of Presocratic Philosophers'},
		                xaxis={'title': 'Philosophers', "tickangle": 80},
		                yaxis={'title': "Schools"},
		                margin={'l': 138, 'b': 150, 't': 50, 'r': 10}
		            )
		        }
		    ),
		])
		    		

	if __name__ == '__main__':
		app.run_server(debug=True)




print (webscraping_csv_visualization('https://en.wikipedia.org/wiki/List_of_ancient_Greek_philosophers'))
print(fragments('Thales.pdf'))
print(wordcloud('figure.figsize'))
print(open_analyze_img('Philosophers/*.jpg'))
print(visualization("https://codepen.io/chriddyp/pen/bWLwgP.css"))
