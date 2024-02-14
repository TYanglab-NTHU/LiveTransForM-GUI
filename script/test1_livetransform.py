import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinterdnd2.TkinterDnD import _require
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk, ImageFilter
from openbabel import openbabel as ob
import sys
sys.path.append('/Users/liuyihsuan/Documents/livetransform/LiveTransForM-main/')
from fast_jtnn.vocab import Vocab
from fast_jtnn.mol_tree import MolTree
from fast_jtnn.util_functions import Find_coordinates
from fast_jtnn.jtprop_vae import JTPropVAE
from tl_SSmodel import PropNN
import os
vocab = os.path.join('../','data','data_vocab.txt')
vocab = [x.strip("\r\n ") for x in open(vocab)]
vocab = Vocab(vocab)


hidden_size_lfs = 450
hidden_size_ss = 56
latent_size = 56
depthT = 20
depthG = 3
prop_size = 2
def __init__(self,vocab,hidden_size_lfs=hidden_size_lfs,hidden_size_ss=hidden_size_ss,latent_size=latent_size,depthT=depthT,depthG=depthG,prop_size=prop_size):
    self.hidden_size_lfs = hidden_size_lfs
    self.hidden_size_ss = hidden_size_ss
    self.latent_size = latent_size
    self.depthT = depthT
    self.depthG = depthG
    self.prop_size = prop_size
    self.vocab = [x.strip("\r\n ") for x in open(vocab)]
    self.vocab = Vocab(self.vocab)
    self._restored_lfs = False
    self._restored_ss = False
    self.zeff_dict = {'Mn2':0.21187286929865048, 'Mn3':0.47965663466222397,
        'Fe2':0.7091855764024294, 'Fe3':0.9769693417660021,
        'Co2':1.2064982835062081, 'Co3':1.474282048869781,
        }
    self.ionization_energy = {'Mn2':0.1489329431875911,'Mn3':1.1052894371297863,
        'Fe2':-0.01555472321511081, 'Fe3':1.3070142633261608,
        'Co2':0.1397790435114532,'Co3':1.1086797703431708,
        }
    self.ionic_radi = {'Mn2':-0.4696827439384181,'Mn3':-0.7082050151547933,
        'Fe2':-0.6286975914160016,'Fe3':-1.0262347101099603,
        'Co2':-0.5226876930976125,'Co3':-0.8009636761833837
        }
    self.ss_dict = {'HS': 1, 'LS': 0, 1: 'HS', 0:'LS'}
        

def restore(self):
    model_lfs_path = '/Users/liuyihsuan/Documents/livetransform/LiveTransForM-main/data/model/JTVAE_model.epoch-89'
    model_lfs = JTPropVAE(vocab, int(hidden_size_lfs), int(latent_size),int(prop_size),int(depthT),int(depthG))
    #dict_buffer = torch.load(model_lfs_path, map_location='cuda:0')
    dict_buffer = torch.load(model_lfs_path, map_location=torch.device('cpu'))
    model_lfs.load_state_dict(dict_buffer)
    #model_lfs.cuda()
    model_lfs.eval()
    self._restored_lfs = True
    self.model_lfs = model_lfs
    model_ss_path = '/Users/liuyihsuan/Documents/livetransform/LiveTransForM-main/data/model/SS_model.epoch-100'
    #dict_buffer = torch.load(model_ss_path, map_location='cuda:0')
    dict_buffer = torch.load(model_ss_path, map_location=torch.device('cpu'))
    model_ss = PropNN(28,56,0.5)
    model_ss.load_state_dict(dict_buffer)
    #model_ss.cuda()
    model_ss.eval()
    self._restored_ss = True
    self.model_ss = model_ss

def smile2zvec(self,smis):
    tree_batch = [MolTree(smi) for smi in smis]
    _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
    tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
    z_tree, z_mol = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
    z_vecs = torch.cat((z_tree,z_mol),dim=1).cuda()
    return z_vecs
    
def obtain_ss_unique(self,axil_smis,equ_vecs,metal):
    restore(self)
    if len(axil_smis) != 2 or len(equ_vecs) != 4:
        return None
    axil_vecs = smile2zvec(self, axil_smis)
    equ_vecs = smile2zvec(self, equ_vecs)
    axil_batch_tensor = torch.stack(tuple(axil_vecs), dim=0).sum(dim=0)
    equ_batch_tensor = torch.stack(tuple(equ_vecs), dim=0).sum(dim=0)
    zeff = torch.tensor(self.zeff_dict[metal])
    ionic_radi = torch.tensor(self.ionic_radi[metal])
    ionization_energy = torch.tensor(self.ionization_energy[metal])
    metal_inform_batch = torch.stack([zeff, ionic_radi, ionization_energy]).to(torch.float32)
    #complex_predict = self.model_ss.ligand_NN(torch.cat((axil_batch_tensor.cuda(),equ_batch_tensor.cuda(),metal_inform_batch.cuda()),dim=0))
    complex_predict = self.model_ss.ligand_NN(torch.cat((axil_batch_tensor,equ_batch_tensor,metal_inform_batch),dim=0))
    _, complex_predict = torch.max(complex_predict,0)
    spin_state = self.ss_dict[complex_predict.item()]
    return spin_state

def IdxTosmiles(self,axial_equ_pair,idx_dict):
    batches = []
    for axial_smiles, equ_smiles in axial_equ_pair.items():
        batch = []
        axial_pair = tuple(idx_dict[key] for key in axial_smiles)
        batch.extend(axial_pair)
        equ_pair = tuple(idx_dict[key] for key in equ_smiles)
        batch.extend(equ_pair)
        batches.append(batch)
    final_batches = [tuple(batch) for batch in batches]  # convert each sub-list to a tuple
    final_batches = list(set(final_batches))  # remove duplicates
    return final_batches

class LiveTransForM(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("LiveTransForM")
        self.master.geometry("800x600")
        self.pack(fill=tk.BOTH, expand=True)
        self.title = tk.Label(self, text="Choose one required function", font="none 18 bold")
        self.title.pack(pady=10)
        self.create_widgets()

    def create_widgets(self):
        self.button1 = tk.Button(self, text="Single Mutation", command=self.show_single_mutation)
        self.button1.pack(pady=10)

        self.button2 = tk.Button(self, text="Seeded Generation", command=self.show_seeded_generation)
        self.button2.pack(pady=10)

        self.button3 = tk.Button(self, text="Predict Spin State", command=self.show_predict_spin_state)
        self.button3.pack(pady=10)

    def show_single_mutation(self):
        self.master.destroy()
        root = TkinterDnD.Tk()
        app = Single_mutation(root)
        app.mainloop()

    def show_seeded_generation(self):
        self.master.destroy()
        root = TkinterDnD.Tk()
        app = Seeded_generation(root)
        app.mainloop()

    def show_predict_spin_state(self):
        self.master.destroy()
        root = TkinterDnD.Tk()
        app = Predict_spin_state(root)
        app.mainloop()

class Single_mutation(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("LiveTransForM")     
        self.master.geometry("800x600")

        # Frame1
        self.singlemuta_frame = tk.Frame(self.master, pady=10)
        #self.singlemuta_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.singlemuta_frame.pack()
        self.title = tk.Label(self.singlemuta_frame, text="Single Mutation", font="none 18 bold")
        #self.title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        self.xyz_file_path = None
        self.create_widgets()
        self.title = tk.Label(self.singlemuta_frame, text=""""Please assign the metal center oxidation state from the following options:
        [Fe2, Fe3, Mn2, Mn3, Co2, Co3]
        You can also type 'exit' to quit: """, font="none 12 bold")
        self.title.pack()
        self.metal_entry = tk.Entry(self.singlemuta_frame, width=10)
        self.metal_entry.pack()
        self.get_button = tk.Button(self.singlemuta_frame, text="確認", command=self.get_metal)
        self.get_button.pack()
        self.output_label = tk.Label(self.singlemuta_frame, text="")
        self.output_label.pack()

        # Frame2
        self.ligands_frame = customtkinter.CTkScrollableFrame(self.master)
        #self.ligands_frame = tk.Frame(self.master, pady=10)
        #self.ligands_frame.pack(pady=20, fill=tk.BOTH, expand=True)
        self.ligands_frame.pack()
        self.title = tk.Label(self.ligands_frame, text="Choose Ligand", font="none 18 bold")
        #self.title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        self.title.pack()
        # Generated Pictures
        #self.diagram_to_button(image="")
        # self.diagram_to_button(image="/Users/liuyihsuan/Documents/livetransform/LiveTransForM-main/script/button.png")
        # self.diagram_to_button(image="/Users/liuyihsuan/Documents/livetransform/LiveTransForM-main/script/button2.png")
        self.mutation_button = tk.Button(self.ligands_frame, text="Perform Single Mutation", command=self.perform_single_mutation)
        self.mutation_button.pack()

        # Frame3
        self.nodes_frame = tk.Frame(self.master, pady=10)
        self.nodes_frame.pack()
        self.title = tk.Label(self.nodes_frame, text="", font="none 18 bold")
        self.title.pack()
        self.entry = tk.Entry(self.nodes_frame, width=10)
        self.entry.pack()
        self.get_button = tk.Button(master, text="確認", command=self.get_nodes)
        self.get_button.pack()

        # Frame4
        self.start_frame = tk.Frame(self.master, pady=10)
        self.start_frame.pack()
        self.get_button = tk.Button(master, text="Start", command=self.get_nodes)
        self.get_button.pack()

        self.output_label_sec = tk.Label(self.master, text="")
        self.output_label_sec.pack()

        self.ligands_output = tk.Label(self.ligands_frame, text="")
        self.ligands_output.pack()

        # Back to LiveTransForM
        self.back = tk.Button(self.master, text="Back", command=self.switch_to_livetransform_frame)
        self.back.place(x=0, y=0)

    def create_widgets(self):
        # Drag and drop files area
        self.drop_box = tk.Listbox(self.singlemuta_frame, selectmode=tk.SINGLE, bd=2, relief="sunken", background="#ECECEC",)
        #self.drop_box.grid(row=1, column=0, pady=10)
        self.drop_box.pack()
        self.drop_box.drop_target_register(DND_FILES)
        self.drop_box.dnd_bind("<<Drop>>", self.drop)
        # Instructional Text
        sentence = """拖曳檔案上傳"""
        self.test = tk.Label(self.drop_box, text=sentence, fg="#5F5F5F", font="none 15 bold", height=5, width=15, anchor="center")
        #self.test.grid(row=2, column=0)
        self.test.pack()
        self.select_file = tk.Button(self.drop_box, text="選擇檔案", command=self.open_selected_file)
        #self.select_file.grid(row=3, column=0, pady=10)
        self.select_file.pack()

    def drop(self, event):
        filepath = event.data
        filepath = filepath.strip('{}')
        self.xyz_file_path = filepath
        print(self.xyz_file_path)

    def open_selected_file(self):
        print("Before opening file dialog")
        filepath = filedialog.askopenfilename(initialdir='/Users/liuyihsuan/Documents/livetranform/LiveTransForM-main/script/', filetypes=[("xyz files", "*.xyz")])
        print("After opening file dialog")
        self.xyz_file_path = filepath

    def get_metal(self):
        metal_input = self.metal_entry.get()

    def Single_mutation_from_xyz(self,xyz_file,metal=False,SS=False,scs_limit=None,step_size=0.01,max_step_size_limit=100):
        valid_oxidation_states = ["Fe2", "Fe3", "Mn2", "Mn3", "Co2", "Co3"]
        obmol = ob.OBMol()
        obConversion = ob.OBConversion()
        obConversion.SetInFormat("xyz")
        obConversion.ReadFile(obmol, xyz_file)
        SS_change = False
        print(metal)
        # if metal is None or metal == '':
        #     self.output_label.config(text="Metal input is empty. Please enter a valid metal or type 'exit' to quit.")
        #     return
        # if metal.lower() == 'exit':
        #     self.output_label.config(text="Exiting the program. Goodbye!")
        #     return
        
        # if metal in valid_oxidation_states:
        #     self.output_label.config(text=f"You've chosen the oxidation state: {metal}")
        #     metal = metal
        #     return
        # else:
        #     self.output_label.config(text="Invalid input. Please choose from the provided options.")
        # if not scs_limit:
        #     while True:
        #         user_input = input("Please assign the SCScore for the mutated ligand (or press enter if you don't want any limit): ")
        #         if user_input.lower() == '':
        #             print("No limit set for the SCScore.")
        #             break
        #         try:
        #             scs_limit = float(user_input)
        #             if 1 <= scs_limit <= 5:
        #                 print(f"SCScore set to: {scs_limit}")
        #                 # You can perform further processing based on the SCScore here
        #                 break
        #             else:
        #                 print("SCScore between 1 ~ 5. Please try again.")
        #         except:
        #             pass
        # Perform single mutation on ligands
        if len(obmol.Separate()) == 1:
            metal_atom_count = 0
            for atom in ob.OBMolAtomIter(obmol):
                if atom.IsMetal():
                    metal_atom_count += 1
            if metal_atom_count != 1:
                raise ValueError('Sorry, the model is not afford to dinulear metal sysyem')
            else:
                axial_equ_pair,denticity_dict,idx_dict = Find_coordinates(obmol)
                print(idx_dict)
                spin_state_prediction = set()  # Initialize a set to store spin state predictions
                final_batches = IdxTosmiles(self,axial_equ_pair,idx_dict)
                for batch in final_batches:
                    axial_smiles = batch[:2]
                    equ_smiles = batch[2:]
                    spin_state = obtain_ss_unique(self, axial_smiles, equ_smiles, metal)
                    print('spin_state is', spin_state)
                    if spin_state != None:
                        spin_state_prediction.add(spin_state)
                if not SS:
                    if len(spin_state_prediction) != 1:
                        raise ValueError('The model predicts different spin states in different orientations')
                    else:
                        SS = next(iter(spin_state_prediction))
                        print('original spin_state is', SS)

                # Print the available ligands for the user to choose from
                print('Which ligand do you want to mutate?')
                for idx, lig in idx_dict.items():
                    self.ligands_output.config(text=f"lig {idx}: {lig}")
    def Single_mutaion(self,ligand_choose):
        

    def perform_single_mutation(self):
        xyz_file = self.xyz_file_path
        metal = self.metal_entry.get()
        SS = None  
        scs_limit = None
        self.Single_mutation_from_xyz(xyz_file, metal, SS, scs_limit)
        self.output_label.config(text="Single mutation completed!")


    def diagram_to_button(self, image, button_y=1):
        image = Image.open(image)
        image = image.resize((50, 50))
        button_image = ImageTk.PhotoImage(image)

        self.diagrams_button = tk.Button(self.ligands_frame, image=button_image, command=self.button_clicked)
        self.diagrams_button.image = button_image  
        self.diagrams_button.pack()

    def button_clicked(self):
        messagebox.showinfo("Button Clicked", "The button was clicked!")

    def get_nodes(self):
        nodes = self.entry.get()
        #print(nodes)
        return nodes

    def switch_to_livetransform_frame(self):
        self.master.destroy()
        root = TkinterDnD.Tk()
        app = LiveTransForM(root)
        app.mainloop()

class Seeded_generation(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Seeded Generation")     
        self.master.geometry("800x600")

        # Frame1
        self.singlemuta_frame = tk.Frame(self.master, pady=10)
        #self.singlemuta_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.singlemuta_frame.pack()
        self.title = tk.Label(self.singlemuta_frame, text="Seeded Generation", font="none 18 bold")
        #self.title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        self.title.pack(pady=5)
        self.create_widgets()

        # Frame2
        self.step_frame = tk.Frame(self.master, pady=10)    
        self.step_frame.pack()  
        self.title = tk.Label(self.step_frame, text="Step", font="none 18 bold")
        self.title.pack()
        self.entry = tk.Entry(self.step_frame, width=10)    
        self.entry.pack()
        self.get_button = tk.Button(master, text="確認", command=self.get_steps)
        self.get_button.pack()

        # Frame3
        self.start_frame = tk.Frame(self.master, pady=10)
        self.start_frame.pack()
        self.get_button = tk.Button(master, text="Start", command=self.get_steps)
        self.get_button.pack()

        # Back to LiveTransForM
        self.back = tk.Button(self.master, text="Back", command=self.switch_to_livetransform_frame)
        self.back.place(x=0, y=0)

    def create_widgets(self):
        # Drag and drop files area
        self.drop_box = tk.Listbox(self.singlemuta_frame, selectmode=tk.SINGLE, bd=2, relief="sunken", background="#ECECEC",)
        #self.drop_box.grid(row=1, column=0, pady=10)
        self.drop_box.pack()
        self.drop_box.drop_target_register(DND_FILES)
        self.drop_box.dnd_bind("<<Drop>>", self.drop)
        # Instructional Text
        sentence = """拖曳檔案上傳"""
        self.test = tk.Label(self.drop_box, text=sentence, fg="#5F5F5F", font="none 15 bold", height=5, width=15, anchor="center")
        #self.test.grid(row=2, column=0)
        self.test.pack()
        self.select_file = tk.Button(self.drop_box, text="選擇檔案", command=self.open_selected_file)
        #self.select_file.grid(row=3, column=0, pady=10)
        self.select_file.pack()

    def drop(self, event):
        filepath = event.data
        filepath = filepath.strip('{}')
        self.upload_file(filepath)

    def open_selected_file(self):
        # Handle file selection using a file dialog
        filepath = filedialog.askopenfilename(initialdir='/Users/liuyihsuan/Documents/livetransform/', filetypes=[("xyz files", "*.xyz")])
        self.upload_file(filepath)

    def upload_file(self, filepath):
        print(filepath)

    def get_steps(self):
        steps = self.entry.get()
        #print(nodes)
        return steps

    def switch_to_livetransform_frame(self):
        self.master.destroy()
        root = TkinterDnD.Tk()
        app = LiveTransForM(root)
        app.mainloop()

class Predict_spin_state(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Predict Spin State")     
        self.master.geometry("800x600")
        
        # Frame1
        self.singlemuta_frame = tk.Frame(self.master, pady=10)
        #self.singlemuta_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.singlemuta_frame.pack()
        self.title = tk.Label(self.singlemuta_frame, text="Predict Spin State", font="none 18 bold")
        #self.title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        self.title.pack(pady=5)
        self.create_widgets()

        # Frame2
        self.start_frame = tk.Frame(self.master, pady=10)
        self.start_frame.pack()
        self.get_button = tk.Button(master, text="Start", command=self.start)
        self.get_button.pack()

        # Back to LiveTransForM
        self.back = tk.Button(master, text="Back", command=self.switch_to_livetransform_frame)
        self.back.place(x=0, y=0)

    def create_widgets(self):
        # Drag and drop files area
        self.drop_box = tk.Listbox(self.singlemuta_frame, selectmode=tk.SINGLE, bd=2, relief="sunken", background="#ECECEC",)
        #self.drop_box.grid(row=1, column=0, pady=10)
        self.drop_box.pack()
        self.drop_box.drop_target_register(DND_FILES)
        self.drop_box.dnd_bind("<<Drop>>", self.drop)
        # Instructional Text
        sentence = """拖曳檔案上傳"""
        self.test = tk.Label(self.drop_box, text=sentence, fg="#5F5F5F", font="none 15 bold", height=5, width=15, anchor="center")
        #self.test.grid(row=2, column=0)
        self.test.pack()
        self.select_file = tk.Button(self.drop_box, text="選擇檔案", command=self.open_selected_file)
        #self.select_file.grid(row=3, column=0, pady=10)
        self.select_file.pack()

    def drop(self, event):
        filepath = event.data
        filepath = filepath.strip('{}')
        self.upload_file(filepath)

    def open_selected_file(self):
        # Handle file selection using a file dialog
        filepath = filedialog.askopenfilename(initialdir='/Users/liuyihsuan/Documents/livetransform/', filetypes=[("xyz files", "*.xyz")])
        self.upload_file(filepath)

    def upload_file(self, filepath):
        print(filepath)
    
    def start(self):
        print("start")

    def switch_to_livetransform_frame(self):
        self.master.destroy()
        root = TkinterDnD.Tk()
        app = LiveTransForM(root)
        app.mainloop()

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app_live_trans = LiveTransForM(root)
    root.mainloop()


import time
# GPU
start_time = time.time()
torch.mps.synchronize()
a = torch.ones(4000,4000, device="mps")
for _ in range(200):
   a +=a
elapsed_time = time.time() - start_time
print( "GPU Time: ", elapsed_time)



