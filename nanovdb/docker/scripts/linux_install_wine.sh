#!/usr/bin/env bash

set -ex

yum -y install python-simplejson
yum -y install python-six

yum -y install epel-release
yum -y install samba-winbind-clients

#install wine-4.0
yum -y install wine

# wine-5.6 has the same errors as wine-4.0 above
#
# this is horrible, but it is the only way I have found to avoid the error:
# 000b:err:module:import_dll Loading library kernelbase.dll (which is needed by L"C:\\windows\\system32\\KERNEL32.dll") failed (error c000007b).
#yum -y install wine
#yum -y groupinstall 'Development Tools'
#yum -y install libX11-devel freetype-devel zlib-devel libxcb-devel libxslt-devel libgcrypt-devel libxml2-devel gnutls-devel libpng-devel libjpeg-turbo-devel libtiff-devel gstreamer-devel dbus-devel fontconfig-devel
#wget https://dl.winehq.org/wine/source/5.x/wine-5.6.tar.xz
#tar -Jxf wine-5.6.tar.xz
#pushd wine-5.6
#./configure  --enable-win64
#make
#make install
#popd

wget https://ftp.gnome.org/pub/GNOME/sources/msitools/0.100/msitools-0.100.tar.xz
tar -Jxf msitools-0.100.tar.xz
yum install -y bison libgsf-devel libgcab1-devel
pushd msitools-0.100
./configure
make 
make install 
popd
