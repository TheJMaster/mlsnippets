from urllib import request
import csv
import json

def get_camera_urls_csv():
    with open('seattle-traffic-cameras.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONE)
        next(reader) # skip the first column header row
        cameras = [Camera(row) for row in reader]
        return cameras

def get_camera_urls():
    with open('urls.json', 'r') as jsonfile:
        urls = json.load(jsonfile)
        return urls

class Camera(object):
    def __init__(self, csvrow):
        self.owner = csvrow[0]
        self.name = csvrow[1]
        self.imgurl = json.loads(csvrow[2])['url']
        self.streamurl = "https://58cc2dce193dd.streamlock.net/live/" + csvrow[3].split('=')[-1] + "/playlist.m3u8"
        self.weburl = csvrow[4]
        self.location = json.loads(",".join(csvrow[7:])[1:-1])

if __name__ == '__main__':
    print(get_camera_urls()[0].streamurl)