import os
import tkinter.messagebox
from tkinter import *
from tkinter import ttk

from pytorch_models import GANS, CNNS

# create a tkinter root window
root = tkinter.Tk()

# root window title and dimension
root.title("Pansharpening Image Fusion")
root.geometry('500x300')


# Network Selection
def selection_type_changed(event):
    selected_type = combo_type_cnn_gan.get()
    # CNN or GAN
    if selected_type == "GAN":
        possible_values = [e.name for e in GANS]
    else:
        possible_values = [e.name for e in CNNS]
    combo_type_net['values'] = possible_values
    label2.config(text=f"Please select the type of the {selected_type}:")
    combo_type_net.grid(row=2, column=2)
    combo_type_model.set("")
    combo_type_model['values'] = []
    combo_type_net.set("")
    pass


# model Selection
def selection_net_changed(event):
    global d
    selected_type = combo_type_net.get()

    base_dir = "pytorch_models/trained_models"
    d = {}
    for fold in os.listdir(base_dir):
        if not os.path.isdir(f"{base_dir}/{fold}"):
            continue
        for net in os.listdir(f"{base_dir}/{fold}"):
            if net.upper() == selected_type:
                d[fold] = os.listdir(f"{base_dir}/{fold}/{net}")

    possible_values = []
    for i in d.values():
        possible_values.extend(i)
    combo_type_model['values'] = possible_values
    label3.config(text=f"Please select the type of the {selected_type}:")
    combo_type_model.grid(row=3, column=2)
    combo_type_model.set("")
    pass


def selection_model_changed(event):
    selected_model = combo_type_model.get()
    global d
    for e in d.items():
        if selected_model in e[1]:
            satellite = e[0]
            break
    label32.config(text=f"Trained on {satellite}. {8 if satellite in ['W2', 'W3'] else 4} bands")


# CNN or GAN
label1 = ttk.Label(text="Please select the type of the network:")
label1.grid(row=1, column=1)
combo_type_cnn_gan = ttk.Combobox(state="readonly", values=["CNN", "GAN"])
combo_type_cnn_gan.bind("<<ComboboxSelected>>", selection_type_changed)
combo_type_cnn_gan.grid(row=1, column=2)

# Which net
label2 = ttk.Label()
label2.grid(row=2, column=1)
combo_type_net = ttk.Combobox(state="readonly")
combo_type_net.bind("<<ComboboxSelected>>", selection_net_changed)

# Which trained model
label3 = ttk.Label()
label3.grid(row=3, column=1)
combo_type_model = ttk.Combobox(state="readonly")
label32 = ttk.Label()
label32.grid(row=3, column=3)
combo_type_model.bind("<<ComboboxSelected>>", selection_model_changed)

# Select Image
label4 = ttk.Label(text="Please select the image to fuse:")
label4.grid(row=5, column=1)
img_names = []
data_dir = "../data/FR/"
for sensor in os.listdir(data_dir):
    for img in os.listdir(f"{data_dir}/{sensor}"):
        img_names.append(img[:-4])

combo_type_image = ttk.Combobox(state="readonly", values=img_names)
combo_type_image.grid(row=5, column=2)


def fuse():
    global d
    img_name = combo_type_image.get()
    satellite_test = img_name[:2]
    filtered = list(filter(lambda n: n[:2] == satellite_test, img_names))
    index = filtered.index(img_name) + 1
    type_model = combo_type_net.get()
    name = combo_type_model.get()
    for e in d.items():
        if name in e[1]:
            satellite = e[0]
            break

    if (satellite in {"W2", "W3"} and satellite_test in {"W4", "GE"}) or \
            (satellite_test in {"W2", "W3"} and satellite in {"W4", "GE"}):
        tkinter.messagebox.showerror("Error", "Testing image with different number of bands")
        return

    os.system(
        f"python ../generate_image.py --index_test {index} --satellite {satellite} --satellite_test {satellite_test}"
        f" -t {type_model} -n {name} -mp ../pytorch_models/trained_models")


button = Button(root, text="Fuse", command=fuse, pady=10)
button.grid(row=6, column=2)

root.mainloop()
