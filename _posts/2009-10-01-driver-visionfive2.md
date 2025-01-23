---
title: 'Add a linux driver on the VisionFive2 board'
date: 2012-08-14
permalink: /posts/2012/08/blog-post-1/
tags:
  - VisionFive2 board
  - Linux kernel
---

Add a custom linux driver on the VisionFive2 board.
======

In this article, we will see how we can introduce a custom driver on the linux kernel of the VisionFive2 board. 

Step 1: setup Debian on the VisionFive2 board.
------

Take your microSD card, download the image and Balena Etcherd and flash it. This step is straightforward. Then you can connect using user / starfive as password

Step 2: Download a specific linux kernel
------

By default, dynamic modules are disabled in the Debian image and we can't compile a kernel modules since the kernel file don't have a build folder. To address this issue, we can download another release that support dynamic modules. For this tutorial, we use the version v3.6.1

```bash
echo "starfive" | sudo -S su
wget https://github.com/starfive-tech/VisionFive2/releases/download/VF2_v3.6.1/Image.gz
cp Image.gz /boot/new_kernel
```


Step 3: Add the new kernel in the boot menu
------

Now that we have the new kernel downloaded on the board, the next step consit of placing the kernel in the /boot folder and add a new entry in the bootloader. 

label l2
        menu label The new kernel 
        linux /new_kernel 
        initrd /initrd.img-5.15.0-starfive

        fdtdir /dtbs
        append  root=/dev/mmcblk1p4 rw console=tty0 console=ttyS0,115200 earlycon rootwait stmmaceth=chain_mode:1 selinux=0

Step 4: Reboot the board
------

Now you can reboot the board, in the bootloader menu press 3. If the kernel doesn't boot manually, make sure the path is correct. Also take care that the kernel is found in the bootloader, otherwise the bootloader will boot the default kernel.

Step 5: Generate the module files
------

On the board, execute the following lines (it will take around two hours). It is important to have the exact same commit as the kernel we downloaded


git clone https://github.com/starfive-tech/linux/ --branch visionfive --single-branch --depth 1
cd linux
git fetch --depth 1 origin 05533e9c31d6f0da20efc2d436a3b0f6d516ed4b
git checkout 05533e9c31d6f0da20efc2d436a3b0f6d516ed4b

make starfive_visionfive2_defconfig
make -j4 all
make modules_install


Step 6: Compiling and inserting the module
------

The last step consist of cloning the example kernel module we use in this tutorial and compile it. As a last step we need to add it in the kernel

echo "starfive" | sudo -S su
git clone ADD URL
make all
insmod module.ko


If you see the output, congratulation! You have a custom (minimal) driver on the visionfive2 board!