#import the necessary packages
from __future__ import print_function
from scipy.spatial import distance as dist
import numpy as np
import cv2
import sys
import argparse
import imutils
from imutils import contours
from imutils import perspective
import sys
import tensorflow as tf
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time

from flask import Flask
from flask_restful import reqparse, Api, Resource
from flask import request
from PIL import Image
from io import BytesIO
from flask import jsonify

import pyimgur
import forex_python
from forex_python import converter
from forex_python.converter import CurrencyRates


app = Flask(__name__)
api = Api(app)

CLIENT_ID = "e74b87a3eaa1678"
im = pyimgur.Imgur(CLIENT_ID)
c = CurrencyRates()

@app.route('/', methods=['GET', 'POST'])
def home_page():
	print(request.method)
	if request.method == "POST":
		try:
			filestr = (request.files['file'].read())
			img = Image.open(BytesIO(filestr))
			print("Image recieved from front end!")
			img.save("/currency_img.jpg")
			immutable_dict_obj = request.form.to_dict()
			incoming_string = (immutable_dict_obj['MultipartEntity'])
			num_notes = int(incoming_string[0])
			currency_req = str(incoming_string[2:])
			print("The number of notes needed is:" + str(num_notes) + "The currency to convert to is:" + currency_req + "------")
			print("Image saved and being sent to backend")
			results = run_main("/currency_img.jpg", num_notes, currency_req)
			print("Have the results! Returning to front end")
			return results
		except Exception as e:
                        print(e)
	print("If statement has been executed - Now running!")
	return '<h1>Welcome to my site!</h1>'

def order_points_old(pts):
	rect = np.zeros((4, 2), dtype="float32")
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def crop_minAreaRect(img, rect):
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]
    return img_crop

def order_points(pts):
	xSorted = pts[np.argsort(pts[:, 0]), :]
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	return np.array([tl, tr, br, bl], dtype="float32")

currency_arr = {"USDs": 0, "SGDs":0, "MYRs":0, "INDs":0}

def calculate_exchange(usds, sgds, myrs, inds, currency_to_convert_to):
	print("Getting exchange rates now!")
	usds_in_sgd = usds*c.get_rate('USD', 'SGD')
	myr_in_sgd = myrs*c.get_rate('MYR', 'SGD')
	inds_in_sgd = inds*c.get_rate('IDR', 'SGD')
	total_in_sgd = usds_in_sgd + myr_in_sgd + +inds_in_sgd + sgds
	total = 0
	if(currency_to_convert_to=="USD"):
		total = total_in_sgd*c.get_rate('SGD', 'USD')
	elif(currency_to_convert_to=="SGD"):
		total = total_in_sgd
	elif(currency_to_convert_to=="MYR"):
		total = total_in_sgd*c.get_rate('SGD', 'MYR')
	print("Got the exchange rates!")
	return (total)

ri = 0
def run_main(img_loc, num_img, currency_req):
	start = time.time()

	image_location = img_loc
	num_images = num_img
	currency_required = currency_req

	image = cv2.imread(image_location)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)
	#gray = cv2.threshold(gray,200,200,cv2.THRESH_BINARY)

	canned = cv2.Canny(gray, 50, 100)
	dilated = cv2.dilate(canned, None, iterations=1)
	edged = cv2.erode(dilated, None, iterations=1)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	(cnts, _) = contours.sort_contours(cnts)

	colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
	curr_index = 1
	# loop over the contours individually
	img_arr = []
	to_ret_img = ''

	al_vals = []
	area_vals = []
	for (i, c) in enumerate(cnts):
		area_vals.append(cv2.arcLength(c, 0))
		(x,y),radius = cv2.minEnclosingCircle(c)
		al_vals.append(radius)

	al_sorted = np.array(al_vals)
	al_sorted = np.sort(al_sorted)

	area_sorted = np.array(area_vals)
	area_sorted = np.sort(area_sorted)

	max_radius = al_sorted[-1]
	max_area = area_sorted[-1]

	legit_vals = []
	for min_legit_val in al_sorted:
		if(min_legit_val < (0.75*max_radius)):
			continue
		else:
			legit_vals.append(min_legit_val)

	print("The arclenght values in the full image are: ")
	print((legit_vals))

	for (i, c) in enumerate(cnts):
		#Then, instead of checking for the absolute value of the arclenght, look for the top n arclengths.
		#if((cv2.arcLength(c,0)<1000) or (len(cv2.approxPolyDP(c, (0.1 * cv2.arcLength(c,0)), True))!=4)):
		(x,y),radius = cv2.minEnclosingCircle(c)
		if(radius not in legit_vals):
			continue
		#(x2,y2),radius2 = cv2.minEnclosingCircle(c)
		#rejected_boxes.append(radius2)
		box = cv2.minAreaRect(c)
		roi = crop_minAreaRect(image, box)
		img_arr.append(roi)

		cv2.imwrite('test-img_result'+str(curr_index)+'.JPG', roi)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
		print("Identified " + str(curr_index) + " object(s) in the image")
		rect = order_points_old(box)
	#	if args["new"] > 0:
		rect = perspective.order_points(box)
		for ((x, y), color) in zip(rect, colors):
			cv2.circle(image, (int(x), int(y)), 5, color, -1)
		#TOODODO:UPDATE WRITE TO PATH

		#HEREEEE
		#print("Lets see if it works")
		#final_image_cumm = im.upload_image(image, title="cumm image")
		#final_image_cumm_link = final_image_cumm.link
		#print("Nope, doesn't work")
		cv2.imwrite("test_image_result_cummulative.jpg", image)

		#to_ret_img = Image.open("test_image_result_cummulative.jpg")
		print("Found the first note, doing the machine learning now.")
		##NEED TO DO MACHINE LEARNING FOR EACH IMAGE HERE

		image_data = tf.gfile.FastGFile('test-img_result'+str(curr_index)+'.JPG', 'rb').read()
		label_lines = [line.rstrip() for line in tf.gfile.GFile("tf_files/retrained_labels.txt")]
		with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
		    graph_def = tf.GraphDef()	## The graph-graph_def is a saved copy of a TensorFlow graph; objektinitialisierung
		    graph_def.ParseFromString(f.read())	#Parse serialized protocol buffer data into variable
		    _ = tf.import_graph_def(graph_def, name='')	# import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor
		with tf.Session() as sess:
			softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
			predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
			top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
			index = 0
			top = ''
			for node_id in top_k:
				human_string = label_lines[node_id]
				if (index==0):
					top = human_string
					amount = int(top[3:])
					if("usd" in top):
						currency_arr["USDs"]+=amount
					elif("sgd" in top):
						currency_arr["SGDs"]+=amount
					elif("myr" in top):
						currency_arr["MYRs"]+=amount
					elif("ind" in top):
						currency_arr["INDs"]+=amount
					print("According to me, that is a: " + top + " note. ")
					index = 1

		curr_index = (curr_index + 1)
		print(currency_arr)
	currency_to_convert_to_input = currency_required
	total_value = round(calculate_exchange(currency_arr["USDs"], currency_arr["SGDs"], currency_arr["MYRs"], currency_arr["INDs"], currency_to_convert_to_input), 2)
	print("Total value in " + currency_required + " at the end is:" + str(total_value))

	end = time.time()
	time_taken = end - start
	time_taken = round(time_taken, 2)

	uploaded_image = im.upload_image("test_image_result_cummulative.jpg", title="Final Image to show!")
	to_ret_link = uploaded_image.link
	#to_ret_link = final_image_cumm_link

	print('\n' + "Number of objects identified:" + str(curr_index-1) + " in " + str(time_taken) + " seconds.")

	to_ret_str = ''
	to_ret_usd = str(0) 
	to_ret_sgd = str(0)
	to_ret_ind = str(0)
	to_ret_myr = str(0)

	if(currency_arr["USDs"]!=0):
		to_ret_usd = (str(currency_arr["USDs"]))
	if(currency_arr["SGDs"]!=0):
		to_ret_sgd = (str(currency_arr["SGDs"]))
	if(currency_arr["INDs"]!=0):
		to_ret_ind = (str(currency_arr["INDs"]))
	if(currency_arr["MYRs"]!=0):
		to_ret_myr = (str(currency_arr["MYRs"]))

	to_ret_str += (currency_to_convert_to_input + " : " + str(total_value))
	#print(to_ret_usd + " " + to_ret_sgd + " " + to_ret_ind + " " + to_ret_myr)
	print("The returned string is: " + to_ret_str)
	to_ret = {"result_image_link": to_ret_link, "result_str":to_ret_str, "usd":to_ret_usd, "sgd":to_ret_sgd, "ind":to_ret_ind, "myr":to_ret_myr}

	currency_arr["USDs"]=0
	currency_arr["SGDs"]=0
	currency_arr["MYRs"]=0
	currency_arr["INDs"]=0

	print(to_ret_str)
	print(to_ret_link)
	return jsonify(to_ret)

if __name__ == '__main__':
#	app.run(host='0.0.0.0', port=5433) 
       #app.run(host='170.252.161.137', port=5000, debug=False)
	app.run(host='0.0.0.0', port=5000, debug=False)
