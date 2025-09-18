1) Downloaded the raw data from https://figshare.com/projects/qtaim-generator/196192 stored in "data_suba" folder
  - train_qm9_qtaim_1205_labelled_corrected.pkl
  - test_qm9_qtaim_1205_labelled_corrected.pkl


2) Scripts to process these .pkl files are in "preprocess_43K_suba" folder


   - File name: filter_qm9_to 43K.py and filter_qm9_to 43K.ipynb for vizualization

   - Choose the correct identifier to filter out the 43K

      Molecule key lives in the column "name" in .pkl files 
      i.e, train and test .pkl files has column: name (example: gdb_48609.xyz)
      my_43k.txt file has a list of ids (example: gdb_48609)  
      
   - Filtering out based the id and creating 43 .pkl files and csv files
      train_qm9_qtaim_1205_labelled_corrected_my43k.pkl
      test_qm9_qtaim_1205_labelled_corrected_my43k.pkl

    - Missing ids: 12 molecules in my_43K.txt cannot be found in the .pkl files
      list of the missing ids can be seen in the jupyter notebook: filter_qm9_to 43K.ipynb

3) Converting .pkl files to LMDB for easier processing

   - For whole qm9:
   - Script: preprocess_43K_suba/make_qm9_lmdb_multi.sh : uses configs_suba/settings_qm9_multi.json to create LMDB files
   - Added sitecustomize.py to handle "is_ordered" error, extrakeys:global ['humo', lumo','gap']
   - Logs: logs_suba/lmdb/

    - For 43K subset:
   - Script: preprocess_43K_suba/make_qm9_lmdb_multi_43k.sh : uses configs_suba/settings_qm9_multi_43k.json to create LMDB files
   - Logs: logs_suba/lmdb/
     

    
      
