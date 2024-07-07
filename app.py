import cv2
from ultralytics import YOLO
import os
import streamlit as st
from PIL import Image
import io
import matplotlib.pyplot as plt


def list_images(image_path):
  images = []
  for filename in os.listdir(image_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.jfif')): 
     
      images.append(os.path.join(image_path, filename)) 
  return images



def predict(model_path, 
            images):          
  
    model = YOLO(model_path)  
       
    results = model(images,verbose=False) #model.predict(file)#,stream=True)   
    boxes = results[0].boxes #result.boxes.cpu().numpy()
    
    return boxes.conf, results[0].plot()#[:,:,-1]
    
            
def main(model_path):
    
    st.title("Bala's Antelope Detector")
    st.warning("The model has been trained on only 200 images over 20 epochs, which is considered trivial in Machine Learning terms.\
               Higher numbers of images and epochs typically yield more satisfactory results. Also, the model has not been thoroughly evaluated using the object detection metrics. \
               Therefore, its accuracy in identifying and localizing objects may be limited. This training setup is intended solely \
               for demonstrating the capabilities of the model and may not reflect its performance under more rigorous conditions.",icon="⚠️")
    
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpeg"],accept_multiple_files=True)
        predict_button = st.button("Predict")
   
   
    
    if uploaded_file is not None:   
        with st.container():
            st.write("Images given for predicting")
            input_cols = create_row_col(uploaded_file)
            for image_index, file in enumerate(uploaded_file):               
                img = Image.open(io.BytesIO(file.read()))
                input_cols[image_index].image(img,caption=file.name,width=200)
          
               
    with st.container():
        st.write("Predicted Images")  
                
    if predict_button:
        with st.spinner("Please wait to complete prediction....."): 
            output_img_cols = create_row_col(uploaded_file)
                
            for image_index, file in enumerate(uploaded_file):
                uploaded_image = Image.open(file)#io.BytesIO(file.read()))              
                boxes_conf, res = predict(model_path,uploaded_image)
                img = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                for i in boxes_conf:
                    if i >0.7:
                        with st.container():
                            output_img_cols[image_index].image(img,caption=file.name,width=200,channels="RGB")
        st.success("prediction completed!")


def create_row_col(uploaded_file):
    columns = 3  # Number of columns in the grid
    # Calculate number of rows needed
    n_rows = (len(uploaded_file) // columns) + (1 if len(uploaded_file) % columns > 0 else 0)           
    rows = [st.container() for _ in range(n_rows)]
    cols_per_row = [r.columns(columns) for r in rows]
    cols = [column for row in cols_per_row for column in row]
    return cols


    
if __name__=='__main__':
    model_path=r'.\best.pt' 
    image_path=r'C:\Users\bchandran\OneDrive - Ventia\Ben Stoner\ComputerVisionDemo\images'
    inf_image_path = r'C:\Users\bchandran\OneDrive - Ventia\Ben Stoner\ComputerVisionDemo\inference images'
    main(model_path)
    
