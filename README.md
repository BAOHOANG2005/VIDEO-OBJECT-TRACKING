# Video Object Tracking using SiamFC

## Overview

This project implements **SiamFC (Siamese Fully Convolutional Network)** for **video object tracking**. SiamFC is a state-of-the-art method that tracks objects in video sequences by learning a **similarity function** between a target object template and candidate regions in subsequent frames.  

Unlike traditional tracking methods, SiamFC **does not require online learning**, making it extremely fast and suitable for real-time applications.

---

## Key Features

- **Real-time Object Tracking:** Tracks target objects across frames efficiently.
- **Siamese Network Architecture:** Learns a similarity function between the initial target and candidate regions.
- **Fully Convolutional:** Enables fast sliding-window search without fully connected layers.
- **Supports Various Object Types:** Works on people, vehicles, and arbitrary objects given an initial bounding box.
- **Easy to Extend:** Modular code allows integration of new datasets and pre-trained models.

---

## Project Structure

