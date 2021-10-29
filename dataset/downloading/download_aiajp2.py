
#This python file is used to download compressed magnetograms data in Jp2 format from helioviewer API
# We download two images per day at 00:00 UT and 12:00 UT
#The api provides nearest image to requested timestamp
# We discard such images if the available image has a difference of more than 6 hours from the requested timestamp
# we download images from 2010-2018

import requests
import datetime
start_date = '2010-12-01 00:00:00'

#File counter
counter = 0

#Start Date
dt = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")

for i in range(15000):
    hours = datetime.timedelta(hours=12)
    dt = dt+ hours
    final_date = str(dt.date()) + 'T' + str(dt.time()) + 'Z'
    if dt.year>=2019:
        break

    #Defining name of downloaded images based on the date and time
    filename = 'AIA.m' + str(dt.year) + '.' +  f'{dt.month:02d}' + '.' + f'{dt.day:02d}' + '_'\
         + f'{dt.hour:02d}' + '.' + f'{dt.minute:02d}' + '.' + f'{dt.second:02d}' + '.jp2'
    file_loc_with_name = '/media/chetraj/FC78EEB778EE6FB6/aia/' + filename

    #Using jpip=True gives the uri of the image which we use to parse time_stamp of available image
    #Detail documentation is provided on api.helioviewer.org
    requestString = "https://helioviewer-api.ias.u-psud.fr//v2/getJP2Image/?date=" + final_date + "&sourceId=9&jpip=true"
    #print(requestString, '-->Requested')

    #Parsing date from the recived uri
    response = requests.get(requestString)
    url = str(response.content)
    url_temp = url.rsplit('/', 1)[-1]
    date_recieved = url_temp.rsplit('__', 1)[0][:-4]
    recieved = datetime.datetime.strptime(date_recieved, "%Y_%m_%d__%H_%M_%S")
    #print(recieved)

    #Now comparing the timestamp of available image and requested image
    #Download only if the difference is less than or equal to 6 hours
    if(recieved-dt<=datetime.timedelta(hours=2) or dt-recieved<=datetime.timedelta(hours=2)):
        #print(dt, recieved)
        #This uri provides access to the actual resource ( i.e., images)
        request_uri = "https://helioviewer-api.ias.u-psud.fr//v2/getJP2Image/?date=" + final_date + "&sourceId=9"
        hmidata = requests.get(request_uri)
        open(file_loc_with_name,'wb').write(hmidata.content)
        print(final_date, '-->Downloaded')
        counter+=1

#Total Files Downloaded
print(counter)