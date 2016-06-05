import urllib.request
import datetime

def validate(date_text):
	try:
		datetime.datetime.strptime(date_text, '%m/%d/%Y')
		return True
	except:
		print("ERROR: Incorrect data format (must be MM/DD/YYYY). Try again.")
		return False

def checkticker(ticker):
	try:
		urllib.request.urlretrieve("http://www.google.com/finance/historical?q=NASDAQ:" + ticker + "&output=csv")
		return True
	except:
		print("ERROR: Invalid Stock Ticker. Try again.")
		return False
		
tickername = input("Enter Stock Ticker: ")
while not checkticker(tickername):
	tickername = input("Enter Stock Ticker: ")

startdate = input("Enter Start Date (MM/DD/YYYY): ")
while not validate(startdate):
	startdate = input("Enter Start Date (MM/DD/YYYY): ")
	
enddate = input("Enter End Date (MM/DD/YYYY): ")
while not validate(enddate):
	enddate = input("Enter End Date (MM/DD/YYYY): ")
	
url = "http://www.google.com/finance/historical?q=NASDAQ:" + tickername + "&startdate=" + startdate + "&enddate=" + enddate + "&output=csv"
try:
	urllib.request.urlretrieve(url, tickername + ".csv")
except urllib.error.HTTPError:
	print("Invalid Stock Ticker.")
	
import csv
f1 = open(tickername + '.csv')
f2 = open(tickername + '.txt', 'w', newline='')
creader = csv.reader(f1)
cwriter = csv.writer(f2)
for cline in creader:
	new_line = [val for col, val in enumerate(cline) if col not in (1,2,3,5)]
	cwriter.writerow(new_line)
f1.close()
f2.close()

lines = open(tickername + '.txt').readlines()
open(tickername + '.txt', 'w').writelines(lines[1:])
	
with open(tickername + '.txt', 'r') as f:
	plaintext = f.read()
plaintext2 = plaintext.replace(',', ' ')
with open(tickername + '.txt', 'w') as f:
	f.write(plaintext2)

import numpy as np
x = np.genfromtxt(tickername + '.txt', dtype='str', usecols=range(0,1))

import datetime as dt
def excel_date(date1):
	date = datetime.datetime.strptime(date1, '%d-%b-%y')
	temp = dt.datetime(1899, 12, 31)
	delta = date - temp
	return int(float(delta.days) + (float(delta.seconds) / 86400)+1.0)

for i in range(0, x.shape[0]):
	x[i] = excel_date(x[i])
x = x.astype(int)
y = np.loadtxt(tickername + '.txt', usecols=range(1,2))
data=np.column_stack((x,y))
np.savetxt('training_data.txt', data, '%5.2f')

with open('training_data.txt', 'r+') as f:
	content = f.read()
	f.seek(0, 0)
	f.write("Inputs: 1 Outputs: 1".rstrip('\r\n') + '\n' + content)
	
import os
os.remove(tickername + '.txt')