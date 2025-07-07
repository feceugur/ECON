#!/bin/bash

output_dir="/var/locally-mounted/myshareddir/Fulden"

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

mkdir -p data/smpl_related/models

# username and password input
echo -e "\nYou need to register at https://icon.is.tue.mpg.de/, according to Installation Instruction."
read -p "Username (ICON):" username
read -p "Password (ICON):" password
username=$(urle $username)
password=$(urle $password)

# SMPL (Male, Female)
echo -e "\nDownloading SMPL..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip&resume=1' -O $output_dir'/smpl_related/models/SMPL_python_v.1.0.0.zip' --no-check-certificate --continue
unzip $output_dir/smpl_related/models/SMPL_python_v.1.0.0.zip -d $output_dir/smpl_related/models
mv $output_dir/smpl_related/models/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl $output_dir/smpl_related/models/smpl/SMPL_FEMALE.pkl
mv $output_dir/smpl_related/models/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl $output_dir/smpl_related/models/smpl/SMPL_MALE.pkl
cd $output_dir/smpl_related/models
rm -rf *.zip __MACOSX smpl/models smpl/smpl_webuser
cd ../../..

# SMPL (Neutral, from SMPLIFY)
echo -e "\nDownloading SMPLify..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&sfile=mpips_smplify_public_v2.zip&resume=1' -O $output_dir'/smpl_related/models/mpips_smplify_public_v2.zip' --no-check-certificate --continue
unzip $output_dir/smpl_related/models/mpips_smplify_public_v2.zip -d $output_dir/smpl_related/models
mv $output_dir/smpl_related/models/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl $output_dir/smpl_related/models/smpl/SMPL_NEUTRAL.pkl
cd $output_dir/smpl_related/models
rm -rf *.zip smplify_public 
cd ../../..

# SMPL-X
echo -e "\nDownloading SMPL-X..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip&resume=1' -O $output_dir'/smpl_related/models/models_smplx_v1_1.zip' --no-check-certificate --continue
unzip $output_dir/smpl_related/models/models_smplx_v1_1.zip -d $output_dir/smpl_related
rm -f $output_dir/smpl_related/models/models_smplx_v1_1.zip

# ECON
echo -e "\nDownloading ECON..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=icon&sfile=econ_data.zip&resume=1' -O $output_dir'/econ_data.zip' --no-check-certificate --continue
cd $output_dir && unzip econ_data.zip
mv smpl_data smpl_related/
rm -f econ_data.zip
cd ..

mkdir -p $output_dir/HPS

# PIXIE
echo -e "\nDownloading PIXIE..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=icon&sfile=HPS/pixie_data.zip&resume=1' -O $output_dir'/HPS/pixie_data.zip' --no-check-certificate --continue
cd $output_dir/HPS && unzip pixie_data.zip
rm -f pixie_data.zip
cd ../..

# PyMAF-X
echo -e "\nDownloading PyMAF-X..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=icon&sfile=HPS/pymafx_data.zip&resume=1' -O $output_dir'/HPS/pymafx_data.zip' --no-check-certificate --continue
cd $output_dir/HPS && unzip pymafx_data.zip
rm -f pymafx_data.zip
#cd ../..





