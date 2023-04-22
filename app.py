from flask import Flask,request,render_template,jsonify
from src.pipline.pradiction_pipline import PredictPipline,CustomData
import pandas as pd
import numpy as np

application = Flask(__name__)
app = application

@app.route("/",methods = ["GET","POST"])
def prediction_datapoint():
    if request.method == "GET":
        return render_template("form.html")


    else:
        data = CustomData(
            Delivery_person_Age = float(request.form.get("Delivery_person_Age"))
            , Delivery_person_Ratings = float(request.form.get("Delivery_person_Ratings"))
            , Restaurant_latitude = float(request.form.get("Restaurant_latitude"))
            , Restaurant_longitude = float(request.form.get("Restaurant_longitude"))
            , Delivery_location_latitude = float(request.form.get("Delivery_location_latitude"))
            , Delivery_location_longitude = float(request.form.get("Delivery_location_longitude"))
            , Weather_conditions = int(request.form.get("Weather_conditions"))
            , Road_traffic_density = int(request.form.get("Road_traffic_density"))
            , Vehicle_condition = int(request.form.get("Vehicle_condition"))
            , Type_of_order = int(request.form.get("Type_of_order"))
            , Type_of_vehicle = int(request.form.get("Type_of_vehicle"))
            , multiple_deliveries = float(request.form.get("multiple_deliveries"))
            , Festival = int(request.form.get("Festival"))
            , City = int(request.form.get("City"))
            )

        final_data = data.get_data_as_data_frame()
        predict_pipline = PredictPipline()
        pred = predict_pipline.predict(final_data)
        result = round(pred[0],2)

        return render_template("form.html",final_result = "Your Delivery Time IS. {}".format(result))


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
