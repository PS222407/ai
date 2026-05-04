```
download from yoda (ibridges)
pip install ibridges
mkdir -p ~/.irods
// paste the config ~/.irods/irods_environment.json from https://geo.yoda.uu.nl/user/data_transfer
ibridges init
// paste password generated from here https://geo.yoda.uu.nl/user/data_access
ibridges pwd // to check if success
ibridges download "irods:~/research-insect-recognizer/T5M7_AT2" .
```
T5M7_AT2


examples:  
Replace sam1.py with sam3.py and owl.py:
```
python3 sam3.py --image C_Users_sina8_Desktop_batch2_T5M7_AT1_2025-07-20_2025-07-20_16-00-53_2025-07-20_16-21-02-646819.jpg --visualize  
python3 sam3.py --image C_Users_sina8_Desktop_batch2_T5M7_AT1_2025-07-20_2025-07-20_10-00-54_2025-07-20_10-01-50-413981.jpg --visualize  
python3 sam3.py --image C_Users_sina8_Desktop_batch2_T5M7_AT1_2025-07-20_2025-07-20_10-00-54_2025-07-20_10-02-04-560922.jpg --visualize  
python3 sam3.py --image test_image.jpg --visualize  
python3 sam3.py --image 2025-07-18_10-05-42-709747.jpg --visualize
```
```commandline
docker compose up -d
```
```commandline
docker compose run insect-detector hf auth login
```
```commandline
docker compose run insect-detector python sam3.py --image-folder ./photos --vis-folder ./results --save-crops ./crops
```
