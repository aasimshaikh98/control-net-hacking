# control-net-hacking

Installation Steps

1. This is the Web UI repo we are using to run Stable Diffusion https://github.com/AUTOMATIC1111/stable-diffusion-webui
- To specifically install this on Apple Silicon (M1/M2) follow these instructions: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon
- The SD model I am using is https://civitai.com/models/28059/icbinp-i-cant-believe-its-not-photography. Place this in the models folder.

2. To run ControlNet, you will need to install the extension here: https://github.com/Mikubill/sd-webui-controlnet

3. To run the Streamlit app
```
streamlit run app.py
```