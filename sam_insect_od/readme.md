examples:  
Replace sam1.py with sam3.py and owl.py:
```
python3 sam3.py --image C_Users_sina8_Desktop_batch2_T5M7_AT1_2025-07-20_2025-07-20_16-00-53_2025-07-20_16-21-02-646819.jpg --visualize  
python3 sam3.py --image C_Users_sina8_Desktop_batch2_T5M7_AT1_2025-07-20_2025-07-20_10-00-54_2025-07-20_10-01-50-413981.jpg --visualize  
python3 sam3.py --image C_Users_sina8_Desktop_batch2_T5M7_AT1_2025-07-20_2025-07-20_10-00-54_2025-07-20_10-02-04-560922.jpg --visualize  
python3 sam3.py --image test_image.jpg --visualize  
python3 sam3.py --image 2025-07-18_10-05-42-709747.jpg --visualize
```

Conclusion
SAM cannot classify, it only segments. You need something on top.

Even claude said:
Your dataset has bounding boxes already labeled, which means you don't need SAM at all. You can train a proper YOLOv8 detector directly on your data. It will learn to detect insects (class 0) and output bounding boxes natively, much cleaner than SAM + filter
