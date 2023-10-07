import streamlit as st
import io
from utils import split_and_combine
from controlnet import ControlnetRequest
import shutil
import os
from PIL import Image
import base64
from faker import Faker

# Create a list to store prompts
if 'prompts' not in st.session_state:
    st.session_state['prompts'] = ["Undersea marine life", "NYC skyline", "Amazon Rainforest", "Anime sword battle"]

# Define Prompts
st.title("Prompts")

with st.form("my_form", clear_on_submit=True):
    new_entry = st.text_input("Enter a prompt to use")
    if st.form_submit_button("Submit"):
        st.session_state['prompts'].insert(0, new_entry)

# Display prompts with remove buttons
for i, entry in enumerate(st.session_state['prompts']):
    col1, col2 = st.columns([4, 1])
    col1.write(entry)
    remove_button = col2.button(f"X", key=f'remove_{i}')
    if remove_button:
        st.session_state['prompts'].pop(i)
        st.rerun()

# Upload Image and Run
st.title("Image Upload and Generation")

uploaded_file = st.file_uploader("Choose a PNG image", type="png")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Parameters to use
    st.write("Enter numerical values for the following parameter. Use a comma if you would like to permutate over multiple values")
    st_steps = st.text_input("Steps", value=20, help="Values between 1-150")
    st_control_weight = st.text_input("Control weight", value=1, help="Values between 0.00-2.00")
    st_starting_control_step = st.text_input("Starting Control Step", value = "0", help="Values between 0.00-1.00")
    st_ending_control_step = st.text_input("Ending Control Step", value = "1", help="Values between 0.00-1.00")
    st_enable_hr = st.checkbox("Enable High Resolution")
    if st_enable_hr:
        st_hr_steps = st.text_input("Hi Res Steps", value="0", help="Values between 1 and 150, Enter 0 to have the same steps as sampler")
    else:
        st_hr_steps = '0'

    if st.button("Start Processing"):
        fake = Faker()
        random_subdirectory_name = f"{fake.word()}-{fake.word()}"
        st.write(f"Images will be saved to the folder `images/{random_subdirectory_name}`")
        random_subdirectory_path = os.path.join(os.getcwd(),'images', random_subdirectory_name)
        os.makedirs(random_subdirectory_path)

        starting_image_path = os.path.join(random_subdirectory_path, "_starting_image.png")
        
        # Copy the uploaded file to the temporary file path
        with open(starting_image_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)
        
        path = starting_image_path
        
        prompts = st.session_state['prompts']

        total_operations = len(st.session_state['prompts'])
        operations_completed = 0

        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0)

        for index, prompt in enumerate(prompts):
            params_to_combine = {'steps': st_steps, "weight": st_control_weight,"guidance_start": st_starting_control_step, "guidance_end": st_ending_control_step, "hr_second_pass_steps":st_hr_steps}
            
            list_of_params_to_run = split_and_combine(params_to_combine)
            for index_2, params in enumerate(list_of_params_to_run):

                control_net = ControlnetRequest(prompt, path)
                control_net.build_body()
                control_net.update_sd({
                    "steps": int(params["steps"]),
                    "enable_hr": st_enable_hr,
                    "hr_second_pass_steps": int(params["hr_second_pass_steps"])}
                    )
                control_net.update_cn({
                        "weight": float(params["weight"]),
                        "guidance_start": float(params["guidance_start"]),
                        "guidance_end": float(params["guidance_end"])
                    }
                )
                output = control_net.send_request()
                result = output['images'][0]

                image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
                gen_image_path = os.path.join(random_subdirectory_path, f"gen_image_{index}_{index_2}.png")
                image.save(gen_image_path)

                st.write(prompt)
                st.write(params)
                st.image(gen_image_path)
                
            # Update the progress bar
            operations_completed += 1
            progress_percent = int((operations_completed / total_operations) * 100)
            my_bar.progress(progress_percent)

            
