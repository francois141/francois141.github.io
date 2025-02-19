---
title: 'Add a linux driver on the VisionFive2 board'
date: 2012-08-14
permalink: /visionfive2-board-custom-kernel/
tags:
  - VisionFive2 board
  - Linux kernel
---
# Add a Custom Linux Driver on the VisionFive2 Board

In this article, we will see how to introduce a custom driver into the Linux kernel of the VisionFive2 board.

## Step 1: Set Up Debian on the VisionFive2 Board

Take your microSD card, download the image and Balena Etcher, and flash it. This step is straightforward. Then you can connect using `user` as the username and `starfive` as the password. Personally I connect either with UART or through SSH.

## Step 2: Download a Specific Linux Kernel

By default, dynamic modules are disabled with the Debian distribution, and we can't compile kernel modules because the kernel sources don't have a build folder. To address this issue, we download another release that supports dynamic modules. For this tutorial, we use version `v3.6.1` and place the new kernel in the boot folder.

```bash
wget https://github.com/starfive-tech/VisionFive2/releases/download/VF2_v3.6.1/Image.gz
sudo cp Image.gz /boot/new_kernel
```

## Step 3: Add the New Kernel to the Boot Menu

Now that we have the new kernel downloaded on the board and located in the boot folder, the next step consists of placing a new entry in the bootloader configuration.

```bash
label l2
    menu label The new kernel
    linux /new_kernel
    initrd /initrd.img-5.15.0-starfive

    fdtdir /dtbs
    append root=/dev/mmcblk1p4 rw console=tty0 console=ttyS0,115200 earlycon rootwait stmmaceth=chain_mode:1 selinux=0
```

## Step 4: Reboot the Board

Now you can reboot the board. In the bootloader menu, press `3`. If the kernel doesn't boot manually, make sure the path is correct. Also, ensure the kernel is found in the bootloader; otherwise, the bootloader will boot the default kernel.

## Step 5: Generate the Module Files

When the step 4 workd, the next step consists of installing the kernel sources to generate the new driver (this will take around two hours). It is important to **use the exact same commit as the kernel we downloaded**.

```bash
git clone https://github.com/starfive-tech/linux/ --branch visionfive --single-branch --depth 1
cd linux
git fetch --depth 1 origin 05533e9c31d6f0da20efc2d436a3b0f6d516ed4b
git checkout 05533e9c31d6f0da20efc2d436a3b0f6d516ed4b

sudo cp /boot/config-5.15.0-starfive .config
make -j4 all
sudo make modules_install
```

## Step 6: Compile and Insert the Module

The last step consists of writing the example kernel module we use in this tutorial and compiling it. The name of the file should be ```driver.c``` As a final step, we need to add it to the kernel.

The driver:

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/uaccess.h>

MODULE_LICENSE("GPL");

static int __init driver_init(void) /* Constructor */
{
    printk(KERN_INFO "Loading driver...\n");

    return 0;
}

static void __exit driver_exit(void) /* Destructor */
{ 
    printk(KERN_INFO "Goodbye from driver!\n");
}

module_init(driver_init);
module_exit(driver_exit);
```

The makefile:

```bash
obj-m += driver.o

all:
	make -C /lib/modules/5.15.0/build M=$(PWD) modules

clean:
	make -C ../linux M=$(PWD) clean
```

```bash
make all
sudo insmod module.ko
sudo dmesg | tail -n 1
```

If you see the output, congratulations! You have successfully added a custom (minimal) driver to the VisionFive2 board!

