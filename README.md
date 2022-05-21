### Flower Classification - Learning Flask, Docker and Postman

##### Original Work in this Repository can be found at the link given below:
https://github.com/DeepthiSudharsan/104-Flower-Classification

This repository has been created with the purpose of learning to use Flask, Docker and Postman. Hence, using a sample saved Mobilenet model and a sample image, the code has been tested out.

##### File Descriptions :

app.py - The modified Python script for this sample project
requirements.txt - Requirements file
0a4ddf5c2.jpeg - Sample image of a waterlily
MobileNet_model.hdf5 - Saved trained MobileNet model
Dockerfile - Docker File
sample.json - Image converted to utf-8 encoded base64 bytes

#### Flask and Postman
-------
##### To run the python file
```
python app.py
```
To pass the image input and check the output returned by the POST method, Postman has been used.

![image](https://user-images.githubusercontent.com/59824729/169663654-0854399c-2e8f-4ee6-94fa-fffa2c26a33e.png)

##### How did we get the value for the json body?

We converted the PIL image to bytes and then utf-8 decoded it. This whole part will happen in the Client side. The output has been copy pasted on the body with the key "myimg" (here).

To convert the image to the decoded format, the following code has been utilized : 

```
img_filename = "0a4ddf5c2.jpeg"
image = Image.open(img_filename)
# Convert to base64
img_byte_arr = io.BytesIO()
image.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()
data = base64.b64encode(img_byte_arr)
with open("sample.json", "w") as outfile:
    outfile.write(data.decode("utf-8"))
```
In the server side, this output string from the requested json needs to be read and converted back to the PIL image. For that purpose, the following code block was written

```
# convert back to image
data = request.get_json()
json_data = data["myimg"]
img_byte_arr = base64.b64decode(json_data)
image = Image.open(io.BytesIO(img_byte_arr))
```
You can see in the above snippet, the output has been printed successfuly after prediction using the saved MobileNet model.

##### How to get the requirements.txt for your python script?
```
pipreqs .
```
Install pipreqs if not installed already. The above command creates a requirements.txt file in the current working directory.

#### Docker
-------
The Dockerfile consists of the following lines of script :
```
FROM python:3.7-slim-buster
WORKDIR /python-docker
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
```
Since I am using python 3.7, so I am starting with python:3.7-slim-buster as the base image.

To build and run, we use the following commands
```
docker build -t flower .
docker run --rm -it -p 5000:5000 flower
```
