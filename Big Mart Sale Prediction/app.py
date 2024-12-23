from flask import Flask, request, render_template
import pandas as pd
import joblib
import os
app = Flask(__name__)

# Load models
xgb_model = joblib.load('XG_model.pkl')
decision_tree_model = joblib.load('DT_model.pkl')

# Mapping dictionaries
item_type_mapping = {
    'Dairy': 0, 'Soft Drinks': 1, 'Meat': 2, 'Fruits and Vegetables': 3,
    'Household': 4, 'Baking Goods': 5, 'Snack Foods': 6, 'Frozen Foods': 7,
    'Breakfast': 8, 'Health and Hygiene': 9, 'Hard Drinks': 10, 'Canned': 11,
    'Breads': 12, 'Starchy Foods': 13, 'Others': 14, 'Seafood': 15
}

item_fat_content_mapping = {'Low Fat': 0, 'Regular': 1}
outlet_size_mapping = {'Small': 0, 'Medium': 1, 'High': 2}
outlet_location_type_mapping = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
outlet_type_mapping = {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}

@app.route('/')
def login():
    return render_template('login.html')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['dataset']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            df = pd.read_csv(file_path)
            df_view = df.head(20) 
            return render_template('preview.html', df_view=df_view)
    return render_template('upload.html')
@app.route('/prediction')
def prediction():
    return render_template("prediction.html")
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    data = request.form
    item_weight = float(data['Weight'])
    item_fat_content = item_fat_content_mapping[data['Item Fat Content']]
    item_visibility = float(data['Item Visibility'])
    item_type = item_type_mapping[data['Item Type']]
    item_mrp = float(data['Item MRP'])
    
    outlet_establishment_year = int(data['Outlet Establishment Year'])
    outlet_size = outlet_size_mapping[data['Outlet Size']]
    outlet_location_type = outlet_location_type_mapping[data['Outlet Location Type']]
    outlet_type = outlet_type_mapping[data['Outlet Type']]

    # Create DataFrame for prediction
    features = pd.DataFrame([[item_weight, item_fat_content, item_visibility, item_type, item_mrp, outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]],
                            columns=['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP',  'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])

    # Get the selected model
    selected_model = data.get('Model')

    if selected_model == 'XGBoost':
        model = xgb_model
    elif selected_model == 'Decision Tree':
        model = decision_tree_model
    else:
        return "Invalid model selected", 400

    # Predict using the selected model
    prediction = model.predict(features)

    # Convert prediction to a native Python type
    output = float(prediction[0])**3

    return render_template('result.html', prediction=round(output, 4), model=selected_model)
@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/evaluation')
def performance():
    return render_template('evaluation.html')
if __name__ == '__main__':
    app.run(debug=True)
