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
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.sping import PIL
import sys
sys.path.append('..')
from fast_jtnn.vocab import Vocab
from fast_jtnn.mol_tree import MolTree
from fast_jtnn.util_functions import Find_coordinates, checksmile
from fast_jtnn.nnutils import create_var
import fast_jtnn.datautils 
from fast_jtnn.jtprop_vae import JTPropVAE
from tl_SSmodel import PropNN
from Find_ss import *
import os
vocab = os.path.join('..','data','data_vocab.txt')
vocab = [x.strip("\r\n ") for x in open(vocab)]
vocab = Vocab(vocab)

class Genlig():
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
        model_lfs_path = '../data/model/JTVAE_model.epoch-89'
        model_lfs = JTPropVAE(self.vocab, int(self.hidden_size_lfs), int(self.latent_size),int(prop_size),int(self.depthT),int(self.depthG))
        #dict_buffer = torch.load(model_lfs_path, map_location='cuda:0')
        dict_buffer = torch.load(model_lfs_path, map_location=torch.device('cpu'))
        model_lfs.load_state_dict(dict_buffer)
        #model_lfs.cuda()
        model_lfs.eval()
        self._restored_lfs = True
        self.model_lfs = model_lfs
        model_ss_path = '../data/model/SS_model.epoch-100'
        #dict_buffer = torch.load(model_ss_path, map_location='cuda:0')
        dict_buffer = torch.load(model_ss_path, map_location=torch.device('cpu'))
        model_ss = PropNN(28,56,0.5)
        model_ss.load_state_dict(dict_buffer)
        #model_ss.cuda()
        model_ss.eval()
        self._restored_ss = True
        self.model_ss = model_ss
    
    def ss_check(self,input_ss):
        try:
            if type(input_ss) == str:
                spinstate = str.upper(input_ss)
                spinstate = self.ss_dict[spinstate]
            if type(input_ss) == int or type(input_ss) == float:
                spinstate = int(input_ss)
        except:
            raise ValueError('Have you assign the spinstate?')
        return spinstate
    
    def get_vector(self,smile=''):
        smi_target = [smile]
        tree_batch = [MolTree(smi) for smi in smi_target]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
        #z_tree_mean = self.model_lfs.T_mean(tree_vecs).cuda()
        #z_mol_mean = self.model_lfs.G_mean(mol_vecs).cuda()
        z_tree_mean = self.model_lfs.T_mean(tree_vecs)
        z_mol_mean = self.model_lfs.G_mean(mol_vecs)
        #z_tree_log_var = -torch.abs(self.model_lfs.T_var(tree_vecs)).cuda()
        #z_mol_log_var = -torch.abs(self.model_lfs.G_var(mol_vecs)).cuda()
        z_tree_log_var = -torch.abs(self.model_lfs.T_var(tree_vecs))
        z_mol_log_var = -torch.abs(self.model_lfs.G_var(mol_vecs))
        return z_tree_mean,z_mol_mean,z_tree_log_var,z_mol_log_var

    def zvec_from_smiles(self,smis):
        self.restore()
        z_vecs_tot = []
        tree_batch = [MolTree(smi) for smi in smis]
        _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
        z_tree, z_mol = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
        z_vecs = torch.cat((z_tree,z_mol),dim=1)
        prop_pred = self.model_lfs.propNN(z_vecs)
        lfs_pred = torch.clamp(prop_pred[0], min=0, max=1)
        lfs_check,scs_check = [prop.item() for prop in lfs_pred],[prop[1].item() for prop in prop_pred]
        denticity_predict = self.model_lfs.denticity_NN(z_vecs)
        _, denticity_predict = torch.max(denticity_predict,1)
        denticity_predict = denticity_predict + 1
        denticity_count = denticity_predict.sum()
        for z_vec,denticity in zip(z_vecs,denticity_predict):
            z_vecs_tot.append([z_vec,denticity])
        return z_vecs_tot,denticity_count.item(),denticity_predict.tolist(),lfs_check,scs_check
    
    def smile2zvec(self,smis):
        tree_batch = [MolTree(smi) for smi in smis]
        _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
        z_tree, z_mol = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
        #z_vecs = torch.cat((z_tree,z_mol),dim=1).cuda()
        z_vecs = torch.cat((z_tree,z_mol),dim=1)
        return z_vecs

    def obtain_ss_unique(self,axil_smis,equ_vecs,metal):
        self.restore()
        if len(axil_smis) != 2 or len(equ_vecs) != 4:
            return None
        axil_vecs = self.smile2zvec(axil_smis)
        equ_vecs = self.smile2zvec(equ_vecs)
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

class Single_mutation(tk.Frame,Genlig):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("LiveTransForM")     
        w, h = self.master.maxsize()
        self.master.geometry("%dx%d" %(w-10, h-10))
        #self.master.geometry("800x600")

        hidden_size_lfs = 450
        hidden_size_ss = 56
        latent_size = 56
        depthT = 20
        depthG = 3
        prop_size = 2
        self.vocab = vocab
        self.hidden_size_lfs = hidden_size_lfs
        self.hidden_size_ss = hidden_size_ss
        self.latent_size = latent_size
        self.depthT = depthT
        self.depthG = depthG    
        self.prop_size = prop_size
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


        # Frame Input
        self.singlemuta_frame = tk.Frame(self.master, pady=10, relief="sunken", borderwidth=2)
        self.singlemuta_frame.pack(side='left', padx=100)
        title_label = tk.Label(self.singlemuta_frame, text="Input", font="Helvetica 18 bold", padx=10)
        title_label.pack(side="top", anchor="w")
        self.create_widgets()
        self.title = tk.Label(self.singlemuta_frame, text="Metal Oxidation", font="Helvetica 18 bold")
        self.title.pack()
        self.instruction_label = tk.Label(self.singlemuta_frame, text="""Please assign the metal center oxidation state from the following options:
        [Fe2, Fe3, Mn2, Mn3, Co2, Co3]
        You can also type 'exit' to quit:""", font="Helvetica 14")
        self.instruction_label.pack()
        self.metal_frame = tk.Frame(self.singlemuta_frame)
        self.metal_frame.pack()
        self.metal_entry = tk.Entry(self.metal_frame, font="Helvetica 14", width=10)
        self.get_button = tk.Button(self.metal_frame, text="確認", command=self.get_metal)
        self.metal_entry.grid(column=0, row=0)
        self.get_button.grid(column=1, row=0)


        # Frame2
        self.ligands_frame = customtkinter.CTkScrollableFrame(self.singlemuta_frame, width=500, height= 300)
        self.ligands_frame.pack()
        self.title = tk.Label(self.ligands_frame, text="Choose Ligand", font="none 18 bold")
        self.title.pack()
        self.mutation_button = tk.Button(self.ligands_frame, text="Perform Single Mutation", command=self.perform_single_mutation)
        self.mutation_button.pack()
        # self.ligands_output = tk.Label(self.ligands_frame, text="")
        # self.ligands_output.pack()

        # Frame3
        self.nodes_frame = tk.Frame(self.singlemuta_frame, pady=10)
        self.nodes_frame.pack()
        self.title = tk.Label(self.nodes_frame, text="Interpolation points", font="none 18 bold")
        self.title.pack()
        self.entry = tk.Entry(self.nodes_frame, width=10)
        self.entry.pack(side=tk.LEFT, padx=5)
        self.get_button = tk.Button(self.nodes_frame, text="確認", command=self.Interpolation_point)
        self.get_button.pack(side=tk.RIGHT, padx=6)

        # Frame4
        self.start_frame = tk.Frame(self.singlemuta_frame, pady=20)
        self.start_frame.pack()
        self.get_button = tk.Button(self.start_frame, text="Start", command=self.Single_mutation)
        self.get_button.pack()
        self.output_label_sec = tk.Label(self.start_frame, text="")
        self.output_label_sec.pack()

        # Frame output
        self.output_frame = customtkinter.CTkScrollableFrame(self.master, width=500, height= 800)
        self.output_frame.pack(side='right', padx=100)
        title_label = tk.Label(self.output_frame, text="Output", font="Helvetica 18 bold", padx=10)
        title_label.pack(side="top", anchor="w")

        # Back to LiveTransForM
        self.back = tk.Button(self.master, text="Back", command=self.switch_to_livetransform_frame)
        self.back.place(x=0, y=0)

    def create_widgets(self):
        self.drop_box = tk.Listbox(self.singlemuta_frame, selectmode=tk.SINGLE, bd=2, relief="sunken", background="#ECECEC",)
        self.drop_box.pack()
        self.drop_box.drop_target_register(DND_FILES)
        self.drop_box.dnd_bind("<<Drop>>", self.drop)
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
        filepath = filedialog.askopenfilename(initialdir='~', filetypes=[("xyz files", "*.xyz")])
        print("After opening file dialog")
        self.xyz_file_path = filepath

    def get_metal(self):
        metal_input = self.metal_entry.get()

    def Choose_ligands_from_xyz(self,xyz_file,metal=False,SS=False,scs_limit=None):
        valid_oxidation_states = ["Fe2", "Fe3", "Mn2", "Mn3", "Co2", "Co3"]
        obmol = ob.OBMol()
        obConversion = ob.OBConversion()
        obConversion.SetInFormat("xyz")
        obConversion.ReadFile(obmol, xyz_file)
        SS_change = False
        print(metal)
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
                self.ligand_images = []
                final_batches = Genlig.IdxTosmiles(self,axial_equ_pair,idx_dict)
                for batch in final_batches:
                    axial_smiles = batch[:2]
                    equ_smiles = batch[2:]
                    spin_state = Genlig.obtain_ss_unique(self, axial_smiles, equ_smiles, metal)
                    #print('spin_state is', spin_state)
                    if spin_state != None:
                        spin_state_prediction.add(spin_state)
                if not SS:
                    if len(spin_state_prediction) != 1:
                        raise ValueError('The model predicts different spin states in different orientations')
                    else:
                        SS = next(iter(spin_state_prediction))
                        #print('original spin_state is', SS)

                # Print the available ligands for the user to choose from
                #print('Which ligand do you want to mutate?')
                for idx, lig in idx_dict.items():
                    mol = Chem.MolFromSmiles(lig)
                    if mol:
                        img_data = Draw.MolToImage(mol)
                        ligand_image_tk = ImageTk.PhotoImage(img_data)
                        self.ligand_images.append(ligand_image_tk)
                #     img_data = Draw.MolToImage(mol)
                #     self.ligands_image_tk = ImageTk.PhotoImage(img_data)
                #     self.mutation_button.config(image=self.ligands_image_tk, compound="top")
                # print(lig_list)
                        
                return self.ligand_images, idx_dict, axial_equ_pair, denticity_dict
    
    def diagram_to_button(self,ligand_images, idx_dict):
        img_data = Image.new('RGB', (10, 10), color='white')
        ligands_image_tk = ImageTk.PhotoImage(img_data)
        for idx, ligand_image in enumerate(ligand_images):
            #button = tk.Button(self.ligands_frame, image=ligand_image, compound="top", command=self.button_clicked)
            button = tk.Button(self.ligands_frame, image=ligand_image, compound="top", command=lambda i=idx: self.button_clicked(i, idx_dict, ligand_images))
            label = tk.Label(self.ligands_frame, text=f"{idx_dict[idx]}")
            button.pack()
            label.pack()

    def button_clicked(self, idx, idx_dict, ligand_images):
        if 0 <= idx < len(idx_dict) and 0 <= idx < len(ligand_images):
            self.selected_smiles = idx_dict[idx]
            self.selected_number = idx

    def perform_single_mutation(self):
        xyz_file = self.xyz_file_path
        metal = self.metal_entry.get()
        SS = None  
        scs_limit = None
        #self.Single_mutation_from_xyz(xyz_file, metal, SS, scs_limit)
        ligand_images, idx_dict, axial_equ_pair, denticity_dict = self.Choose_ligands_from_xyz(xyz_file, metal)
        self.selected_smiles = None
        self.selected_number = None
        self.diagram_to_button(ligand_images, idx_dict)
        #print(f"Using selected SMILES in perform_single_mutation: {self.selected_smiles}")

    def Interpolation_point(self):
        interpolation = self.entry.get()

        return interpolation

    def Single_mutation(self, ligand_choose=None, idx=None, SS=False, scs_limit=None, step_size=0.01, step_size_limit=0, max_step_size_limit=100):
        SS_change = False
        metal = self.metal_entry.get()
        xyz_file = self.xyz_file_path
        ligand_images, idx_dict, axial_equ_pair, denticity_dict = self.Choose_ligands_from_xyz(xyz_file, metal)
        choice = self.selected_number
        selected_ligand = self.selected_smiles
        self.output_images = []
        #print(selected_ligand)
        if selected_ligand:
            step_size_limit = 0
            while not SS_change:
                count = 0
                while count <= 10:
                    idx_dict_copy = idx_dict.copy()
                    if count == 10:
                                count = 0
                                step_size += 0.1
                                step_size_limit += 1
                                print('Increase step size!')
                    if step_size_limit > max_step_size_limit:
                                raise ValueError("Couldn't find a suitable mutated ligand, maybe try a different initial step size")
                    count += 1
                    z_tree_mean, z_mol_mean, z_tree_log_var, z_mol_log_var = self.get_vector(selected_ligand)
                    epsilon_tree = create_var(torch.randn_like(z_tree_mean))
                    epsilon_mol = create_var(torch.randn_like(z_mol_mean))
                    z_tree_mean_new = z_tree_mean + torch.exp(z_tree_log_var / 2) * epsilon_tree * step_size
                    z_mol_mean_new = z_mol_mean + torch.exp(z_mol_log_var / 2) * epsilon_mol * step_size
                    smi_new = self.model_lfs.decode(z_tree_mean_new, z_mol_mean_new, prob_decode=False)
                    smi_new = checksmile(smi_new)
                    if smi_new != checksmile(selected_ligand):
                        # Test decode smiles denticity
                        try:
                            tree_batch = [MolTree(smi_new)]
                            _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
                            tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
                            z_tree_, z_mol_ = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
                            z_vecs_ = torch.cat((z_tree_,z_mol_),dim=1)
                            lfs_pred,scs_pred = self.model_lfs.propNN(z_vecs_).squeeze(0)
                            lfs_pred = torch.clamp(lfs_pred, min=0, max=1).item()
                            scs_pred = torch.clamp(scs_pred, min=1, max=5).item()
                            denticity_predict_check = self.model_lfs.denticity_NN(z_vecs_)
                            if scs_limit:
                                if scs_pred <= scs_limit:
                                    _, denticity_predict_check = torch.max(denticity_predict_check,1)
                                    denticity_predict_check = (denticity_predict_check + 1).item()
                                    if denticity_predict_check == denticity_dict[selected_ligand]:
                                        idx_dict_copy[choice] = smi_new
                                        spin_state_prediction = set()  # Initialize a set to store spin state predictions
                                        final_batches = self.IdxTosmiles(axial_equ_pair,idx_dict_copy)
                                        for batch in final_batches:
                                            axial_smiles = batch[:2]
                                            equ_smiles = batch[2:]
                                            spin_state = self.obtain_ss_unique(axial_smiles, equ_smiles, metal)
                                            if spin_state != None:
                                                spin_state_prediction.add(spin_state)
                                        if len(spin_state_prediction) == 1:
                                            SS_new = next(iter(spin_state_prediction))
                                            if SS != SS_new:
                                                for idx, lig in idx_dict_copy.items():
                                                    print(f"lig {idx}: {lig}")
                                                SS_change = True
                                                break
                            else:
                                _, denticity_predict_check = torch.max(denticity_predict_check,1)
                                denticity_predict_check = (denticity_predict_check + 1).item()
                                if denticity_predict_check == denticity_dict[selected_ligand]:
                                    idx_dict_copy[choice] = smi_new
                                    spin_state_prediction = set()  # Initialize a set to store spin state predictions
                                    final_batches = self.IdxTosmiles(axial_equ_pair,idx_dict_copy)
                                    for batch in final_batches:
                                        axial_smiles = batch[:2]
                                        equ_smiles = batch[2:]
                                        spin_state = self.obtain_ss_unique(axial_smiles, equ_smiles, metal)
                                        if spin_state != None:
                                            spin_state_prediction.add(spin_state)
                                    if len(spin_state_prediction) == 1:
                                        SS_new = next(iter(spin_state_prediction))
                                        if SS != SS_new:
                                            for idx, lig in idx_dict_copy.items():
                                                print(f"lig {idx}: {lig}")
                                            SS_change = True
                                            break
                        except:
                            pass
            final_point = idx_dict_copy[choice]
            inital_point = idx_dict[choice]
            interpolation = self.Interpolation_point()
            while True:
                user_input = self.Interpolation_point()
                if user_input.lower() == '' or user_input.lower() == 'exit':
                    interpolation = False
                    break
                try:
                    delta_step = int(user_input)
                    print(f"Number of interpolation set to: {delta_step}")
                    interpolation = True
                    break
                except:
                    pass
            if interpolation:
                print('Interpolation start')
                mutation_list = []
                inital_vecs = self.zvec_from_smiles([inital_point])
                final_vecs = self.zvec_from_smiles([final_point])
                denticity_ = denticity_dict[selected_ligand]
                zvecs_inital = inital_vecs[0][0][0]
                zvecs_final = final_vecs[0][0][0]
                delta = zvecs_final - zvecs_inital
                one_piece = delta / delta_step
                for i in range(delta_step):
                    idx = 0
                    stepsize = 0.05
                    flag = True
                    zvecs = zvecs_inital + one_piece * (i + 1)
                    while flag:
                        if idx == 0 or idx == 10:
                            idx = 0
                            stepsize += 0.1
                        try:
                            smi = self.model_lfs.decode(*torch.split((zvecs).unsqueeze(0),28,dim=1), prob_decode=False)
                            tree_batch = [MolTree(smi)]
                            _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
                            tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
                            z_tree_, z_mol_ = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
                            z_vecs_ = torch.cat((z_tree_,z_mol_),dim=1)
                            denticity_predict_check = self.model_lfs.denticity_NN(z_vecs_)
                            _, denticity_predict_check = torch.max(denticity_predict_check,1)
                            denticity_predict_check = (denticity_predict_check + 1).item()
                            smi_check = checksmile(smi)
                            if denticity_predict_check != denticity_ or checksmile(smi) == checksmile(inital_point) or checksmile(smi) == checksmile(final_point):
                                zvecs_mutated = z_vecs_ + torch.randn_like(z_vecs_) * stepsize
                                smi = self.model_lfs.decode(*torch.split((zvecs_mutated),28,dim=1), prob_decode=False)
                                tree_batch = [MolTree(smi)]
                                _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
                                tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
                                z_tree_, z_mol_ = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
                                z_vecs_ = torch.cat((z_tree_,z_mol_),dim=1)
                                denticity_predict_check = self.model_lfs.denticity_NN(z_vecs_)
                                _, denticity_predict_check = torch.max(denticity_predict_check,1)
                                denticity_predict_check = (denticity_predict_check + 1).item()
                                smi_check = checksmile(smi)
                                if denticity_predict_check == denticity_ and smi_check != checksmile(inital_point) and smi_check != checksmile(final_point):
                                    flag = False
                                    mutation_list.append([smi_check,i+1])
                            elif denticity_predict_check == denticity_ and smi_check != checksmile(inital_point) and smi_check != checksmile(final_point):
                                flag = False
                                mutation_list.append([smi_check,i+1])
                            idx += 1 
                        except Exception as e:
                            print(e)
                            pass
                print('Interpolation finished')
                for i in mutation_list:
                    smiles,idx = i
                    mol = Chem.MolFromSmiles(smiles)
                    img_data = Draw.MolToImage(mol)
                    ligand_image_tk = ImageTk.PhotoImage(img_data)
                    self.output_images.append(ligand_image_tk)   
                    dig_label = tk.Label(self.output_frame, image=ligand_image_tk)   
                    label = tk.Label(self.output_frame, text=f"{smiles}")
                    dig_label.pack()    
                    label.pack()    
                    print("Point %s: %s" %(idx,smiles))      


    def switch_to_livetransform_frame(self):
        self.master.destroy()
        root = TkinterDnD.Tk()
        app = LiveTransForM(root)
        app.mainloop()


class Seeded_generation(tk.Frame,Genlig):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("LiveTransForM")  
        w, h = self.master.maxsize()
        self.master.geometry("%dx%d" %(w, h))
        #self.master.geometry("800x600")

        hidden_size_lfs = 450
        hidden_size_ss = 56
        latent_size = 56
        depthT = 20
        depthG = 3
        prop_size = 2
        self.vocab = vocab
        self.hidden_size_lfs = hidden_size_lfs
        self.hidden_size_ss = hidden_size_ss
        self.latent_size = latent_size
        self.depthT = depthT
        self.depthG = depthG    
        self.prop_size = prop_size
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

        # Frame Input 
        self.seeded_frame = tk.Frame(self.master, pady=10, relief="sunken", borderwidth=2)
        self.seeded_frame.pack(side='left', padx=100)
        # title_label = tk.Label(self.seeded_frame, text="Input", font="Helvetica 18 bold", padx=5)
        # title_label.pack(side="top", fill="x")
        title_label = tk.Label(self.seeded_frame, text="Input", font="Helvetica 18 bold", padx=10)
        title_label.pack(side="top", anchor="w")
        self.create_widgets()
        self.title = tk.Label(self.seeded_frame, text="Metal Oxidation", font="Helvetica 18 bold")
        self.title.pack()
        self.instruction_label = tk.Label(self.seeded_frame, text="""Please assign the metal center oxidation state from the following options:
        [Fe2, Fe3, Mn2, Mn3, Co2, Co3]
        You can also type 'exit' to quit:""", font="Helvetica 14")
        self.instruction_label.pack()
        self.metal_frame = tk.Frame(self.seeded_frame)
        self.metal_frame.pack()
        self.metal_entry = tk.Entry(self.metal_frame, font="Helvetica 14", width=10)
        self.get_button = tk.Button(self.metal_frame, text="確認", command=self.get_metal)
        self.metal_entry.grid(column=0, row=0)
        self.get_button.grid(column=1, row=0)
        # self.metal_entry.place(relx=0.5, rely=0.81, anchor=tk.CENTER)
        # self.get_button.place(relx=0.75, rely=0.8, anchor=tk.CENTER)
        self.get_button_start = tk.Button(self.seeded_frame, text="Start", command=self.perform_seeded_generation)
        self.get_button_start.pack(anchor=tk.S)

        # #Frame 1
        # self.seeded_frame = tk.Frame(self.master, pady=10)
        # self.seeded_frame.pack()
        # self.create_widgets()

        # Frame metal oxidation
        # self.metal_frame = tk.Frame(self.master, pady=10, relief="sunken", borderwidth=2)
        # self.metal_frame.pack() 
        # self.title = tk.Label(self.metal_frame, text="Metal Oxidation", font="Helvetica 18 bold")
        # self.title.pack()
        # self.instruction_label = tk.Label(self.metal_frame, text="""Please assign the metal center oxidation state from the following options:
        # [Fe2, Fe3, Mn2, Mn3, Co2, Co3]
        # You can also type 'exit' to quit:""", font="Helvetica 14")
        # self.instruction_label.pack()
        # self.metal_entry = tk.Entry(self.metal_frame, font="Helvetica 14", width=10)
        # self.get_button = tk.Button(self.metal_frame, text="確認", command=self.get_metal)
        # self.metal_entry.place(relx=0.5, rely=0.81, anchor=tk.CENTER)
        # self.get_button.place(relx=0.75, rely=0.8, anchor=tk.CENTER)
        # self.output_label = tk.Label(self.metal_frame, font="Helvetica 14")
        # self.output_label.pack(pady=10)

        # # Frame start
        # self.start_frame = tk.Frame(self.master, pady=10)
        # self.start_frame.pack()
        # self.get_button = tk.Button(self.master, text="Start", command=self.perform_seeded_generation)
        # self.get_button.pack()

        # Frame output
        self.output_frame = customtkinter.CTkScrollableFrame(self.master, width=500, height= 800)
        self.output_frame.pack(side='right', padx=100)
        title_label = tk.Label(self.output_frame, text="Output", font="Helvetica 18 bold", padx=10)
        title_label.pack(side="top", anchor="w")
        # self.muta_output = tk.Label(self.output_frame, text="")
        # self.muta_output.pack()
        self.SS_output_label = tk.Label(self.output_frame, text="")
        self.SS_output_label.pack()

        # Back to LiveTransForM
        self.back = tk.Button(self.master, text="Back", command=self.switch_to_livetransform_frame)
        self.back.place(x=0, y=0)

    def create_widgets(self):
        self.drop_box = tk.Listbox(self.seeded_frame, selectmode=tk.SINGLE, bd=2, relief="sunken", background="#ECECEC",)
        self.drop_box.pack()
        self.drop_box.drop_target_register(DND_FILES)
        self.drop_box.dnd_bind("<<Drop>>", self.drop)
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
        filepath = filedialog.askopenfilename(initialdir='~', filetypes=[("xyz files", "*.xyz")])
        print("After opening file dialog")
        self.xyz_file_path = filepath     
        print(self.xyz_file_path)                  

    def get_metal(self):
        metal_input = self.metal_entry.get()

    def Seeded_generation_method(self,xyz_file,metal=False,SS=False,scs_limit=None,step_size=0.01,max_step_size_limit=100):
        xyz_file = self.xyz_file_path
        metal = self.metal_entry.get()
        SS = None  
        scs_limit = None
        obmol = ob.OBMol()
        obConversion = ob.OBConversion()
        obConversion.SetInFormat("xyz")
        obConversion.ReadFile(obmol, xyz_file)
        self.ori_ligand_images = []
        self.mut_ligand_images = []
        if len(obmol.Separate()) == 1:
            metal_atom_count = 0
            for atom in ob.OBMolAtomIter(obmol):
                if atom.IsMetal():
                    metal_atom_count += 1
            if metal_atom_count != 1:
                raise ValueError('Sorry, the model is not afford to dinulear metal sysyem')
            else:
                axial_equ_pair,denticity_dict,idx_dict = Find_coordinates(obmol)
                spin_state_prediction = set()  # Initialize a set to store spin state predictions
                final_batches = self.IdxTosmiles(axial_equ_pair,idx_dict)
                for batch in final_batches:
                    axial_smiles = batch[:2]
                    equ_smiles = batch[2:]
                    spin_state = self.obtain_ss_unique(axial_smiles, equ_smiles, metal)
                    if spin_state != None:
                        spin_state_prediction.add(spin_state)

                print('Seed mutation start')
                idx_dict_copy = idx_dict.copy()
                for idx, lig in idx_dict.items():
                    print(f"lig {idx}: {lig}, Mutation start!")
                    step_size_limit = 0
                    count = 0
                    while count <= 10:
                        # change step size in order to generate diverse ligand
                        if count == 10:
                            count = 0
                            step_size += 0.1
                            step_size_limit += 1
                            print('Increase step size!')
                            print(step_size_limit)
                        if step_size_limit > max_step_size_limit:
                            raise ValueError("Couldn't find a suitable mutated ligand, maybe try a different initial step size")
                        count += 1
                        z_tree_mean, z_mol_mean, z_tree_log_var, z_mol_log_var = self.get_vector(lig)
                        epsilon_tree = create_var(torch.randn_like(z_tree_mean))
                        epsilon_mol = create_var(torch.randn_like(z_mol_mean))
                        z_tree_mean_new = z_tree_mean + torch.exp(z_tree_log_var / 2) * epsilon_tree * step_size
                        z_mol_mean_new = z_mol_mean + torch.exp(z_mol_log_var / 2) * epsilon_mol * step_size
                        smi_new = self.model_lfs.decode(z_tree_mean_new, z_mol_mean_new, prob_decode=False)
                        smi_new = checksmile(smi_new)
                        if smi_new != checksmile(lig):
                            # Test decode smiles denticity
                            try:
                                tree_batch = [MolTree(smi_new)]
                                _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
                                tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
                                z_tree_, z_mol_ = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
                                z_vecs_ = torch.cat((z_tree_,z_mol_),dim=1)
                                lfs_pred,scs_pred = self.model_lfs.propNN(z_vecs_).squeeze(0)
                                lfs_pred = torch.clamp(lfs_pred, min=0, max=1).item()
                                scs_pred = torch.clamp(scs_pred, min=1, max=5).item()
                                denticity_predict_check = self.model_lfs.denticity_NN(z_vecs_)
                                if scs_limit:
                                    if scs_pred <= scs_limit:
                                        _, denticity_predict_check = torch.max(denticity_predict_check,1)
                                        denticity_predict_check = (denticity_predict_check + 1).item()
                                        if denticity_predict_check == denticity_dict[lig]:
                                            idx_dict_copy[idx] = smi_new
                                            break
                                else:
                                    _, denticity_predict_check = torch.max(denticity_predict_check,1)
                                    denticity_predict_check = (denticity_predict_check + 1).item()
                                    if denticity_predict_check == denticity_dict[lig]:
                                        idx_dict_copy[idx] = smi_new
                                        break
                            except:
                                pass
                print('\nMutation Finished \n___________________________\n')
                comparsion_ligs = [(i, j) for i, j in zip(idx_dict.values(), idx_dict_copy.values())]
                for idx, (original_lig, mutated_lig) in enumerate(comparsion_ligs):
                    mol = Chem.MolFromSmiles(original_lig)
                    mutated_mol = Chem.MolFromSmiles(mutated_lig)
                    if mol:
                        img_data = Draw.MolToImage(mol)
                        ligand_image_tk = ImageTk.PhotoImage(img_data)
                        self.ori_ligand_images.append(ligand_image_tk)
                        label = tk.Label(self.output_frame, text=f"{original_lig} -> {mutated_lig}")
                        label.pack()
                        dig_label = tk.Label(self.output_frame, image=ligand_image_tk)
                        dig_label.pack()
                        if mutated_mol:
                            img_data = Draw.MolToImage(mutated_mol)
                            ligand_image_tk = ImageTk.PhotoImage(img_data)
                            self.mut_ligand_images.append(ligand_image_tk)
                            dig_label = tk.Label(self.output_frame, image=ligand_image_tk)
                            dig_label.pack()
                    # self.muta_output.config(text=f'Lig{idx} : Original: {original_lig}, Mutated: {mutated_lig}')
                    
                spin_state_prediction = set()  # Initialize a set to store spin state predictions
                final_batches = self.IdxTosmiles(axial_equ_pair,idx_dict)
                for batch in final_batches:
                    axial_smiles = batch[:2]
                    equ_smiles = batch[2:]
                    spin_state = self.obtain_ss_unique(axial_smiles, equ_smiles, metal)
                    if spin_state != None:
                        spin_state_prediction.add(spin_state)
                if len(spin_state_prediction) != 1:
                    raise ValueError('The model predicts different spin states in different orientations')
                else:
                    SS = next(iter(spin_state_prediction))
                    self.SS_output_label.config(text=f"SS after mutation is {SS}")
                    print('\nSS after mutation is', SS)

            
    def perform_seeded_generation(self):
        xyz_file = self.xyz_file_path
        metal = self.metal_entry.get()
        SS = None  
        scs_limit = None
        self.Seeded_generation_method(xyz_file, metal)

    def switch_to_livetransform_frame(self):
        self.master.destroy()
        root = TkinterDnD.Tk()
        app = LiveTransForM(root)
        app.mainloop()


class Predict_spin_state(tk.Frame,Genlig):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("LiveTransForM")     
        self.master.geometry("800x600")

        hidden_size_lfs = 450
        hidden_size_ss = 56
        latent_size = 56
        depthT = 20
        depthG = 3
        prop_size = 2
        self.vocab = vocab
        self.hidden_size_lfs = hidden_size_lfs
        self.hidden_size_ss = hidden_size_ss
        self.latent_size = latent_size
        self.depthT = depthT
        self.depthG = depthG    
        self.prop_size = prop_size
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

        #Frame Input
        self.predict_frame = tk.Frame(self.master, pady=10, relief="sunken", borderwidth=2)
        self.predict_frame.pack()
        title_label = tk.Label(self.predict_frame, text="Input", font="Helvetica 18 bold", padx=10)
        title_label.pack(side="top", anchor="w")
        self.create_widgets()        
        self.title = tk.Label(self.predict_frame, text="Metal Oxidation", font="Helvetica 18 bold")
        self.title.pack()
        self.instruction_label = tk.Label(self.predict_frame, text="""Please assign the metal center oxidation state from the following options:
        [Fe2, Fe3, Mn2, Mn3, Co2, Co3]
        You can also type 'exit' to quit:""", font="Helvetica 14")
        self.instruction_label.pack()
        self.metal_frame = tk.Frame(self.predict_frame)
        self.metal_frame.pack()
        self.metal_entry = tk.Entry(self.metal_frame, font="Helvetica 14", width=10)
        self.get_button = tk.Button(self.metal_frame, text="確認", command=self.get_metal)
        self.metal_entry.grid(column=0, row=0)
        self.get_button.grid(column=1, row=0)
        # self.metal_entry.place(relx=0.5, rely=0.81, anchor=tk.CENTER)
        # self.get_button.place(relx=0.75, rely=0.8, anchor=tk.CENTER)
        self.get_button_start = tk.Button(self.predict_frame, text="Start", command=self.perform_pred_SS)
        self.get_button_start.pack(anchor=tk.S)


        # Frame output
        self.output_frame = tk.Frame(self.master)
        self.output_frame.pack()
        title_label = tk.Label(self.output_frame, text="Output", font="Helvetica 18 bold", padx=10)
        title_label.pack()


        # Back to LiveTransForM
        self.back = tk.Button(self.master, text="Back", command=self.switch_to_livetransform_frame)
        self.back.place(x=0, y=0)

    def create_widgets(self):
        self.drop_box = tk.Listbox(self.predict_frame, selectmode=tk.SINGLE, bd=2, relief="sunken", background="#ECECEC",)
        self.drop_box.pack()
        self.drop_box.drop_target_register(DND_FILES)
        self.drop_box.dnd_bind("<<Drop>>", self.drop)
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
        filepath = filedialog.askopenfilename(initialdir='~', filetypes=[("xyz files", "*.xyz")])
        print("After opening file dialog")
        self.xyz_file_path = filepath     
        print(self.xyz_file_path)                  

    def get_metal(self):
        metal_input = self.metal_entry.get()

    def Pred_spin_state(self,xyz_file,metal=False):
        obmol = ob.OBMol()
        obConversion = ob.OBConversion()
        obConversion.SetInFormat("xyz")
        obConversion.ReadFile(obmol, xyz_file)
        metal_atom_count = 0
        spin_state_prediction = set()
        if len(obmol.Separate()) == 1:
            for atom in ob.OBMolAtomIter(obmol):
                if atom.IsMetal():
                    metal_atom_count += 1
            if metal_atom_count != 1:
                raise ValueError('Sorry, the model is not afford to dinulear metal sysyem')
            else:
                axial_equ_pair,denticity_dict,idx_dict = Find_coordinates(obmol)
                spin_state_prediction = set()  # Initialize a set to store spin state predictions
                final_batches = self.IdxTosmiles(axial_equ_pair,idx_dict)
                for batch in final_batches:
                    axial_smiles = batch[:2]
                    equ_smiles = batch[2:]
                    spin_state = self.obtain_ss_unique(axial_smiles, equ_smiles, metal)
                    if spin_state != None:
                        spin_state_prediction.add(spin_state)
                if len(spin_state_prediction) != 1:
                    output_label = tk.Label(self.output_frame, text="The model predicts different spin states in different orientations")
                    output_label.pack() 
                    raise ('The model predicts different spin states in different orientations')
                else:
                    output_label = tk.Label(self.output_frame, text=f"The model predict SS is {spin_state}")
                    output_label.pack()
                    print('The model predict SS is %s' %spin_state)
        else:
            output_label = tk.Label(self.output_frame_frame, text="Structure might not be octahedral")
            output_label.pack()
            raise ValueError('Structure might not be octahedral')
    
    def perform_pred_SS(self):
        xyz_file = self.xyz_file_path
        metal = self.metal_entry.get()
        self.Pred_spin_state(xyz_file, metal)

    def switch_to_livetransform_frame(self):
        self.master.destroy()
        root = TkinterDnD.Tk()
        app = LiveTransForM(root)
        app.mainloop()

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app_live_trans = LiveTransForM(root)
    root.mainloop()