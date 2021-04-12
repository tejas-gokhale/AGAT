from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt


##### B, JT, TTT, Ours

### WEATHER
barWidth = 0.2


# b1 = [100-(14.3+13.4+9.2)/3, 100-(21.7+18.4+9.8)/3, 100-(19.0+25.3+10.8)/3, 100-(21.3+26.9+13.3)/3, 70.6]
# b2 = [100-(13.1+12.3+8.4)/3, 100-(21.2+17.5+9.4)/3, 100-(18.4+25.0+11.4)/3, 100-(21.1+25.4+14.1)/3, 71.7]
# b3 = [100-(12.8+11.9+8.2)/3, 100-(20.2+16.9+9.2)/3, 100-(17.8+23.3+11.0)/3, 100-(20.1+24.0+13.5)/3, 73.7]
# b4 = [86.3, 83.4, 81.4, 79.6, 77.8]


# # Set position of bar on X axis
# r1 = np.arange(len(b1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
# r4 = [x + barWidth for x in r3]
# r5 = [x + barWidth for x in r4]


# # Make the plot
# plt.bar(r1, b1, color='r', width=barWidth, edgecolor='white', label='B+SS')
# plt.bar(r2, b2, color='y', width=barWidth, edgecolor='white', label='JT')
# plt.bar(r3, b3, color='g', width=barWidth, edgecolor='white', label='TTT')
# plt.bar(r4, b4, color='b', width=barWidth, edgecolor='white', label='Ours')
# # plt.bar(r5, L5, color='#007f5e', width=barWidth, edgecolor='white', label='var5')
 
# # Add xticks on the middle of the group L
# plt.xlabel('Weather', fontweight='bold', fontsize=18)
# plt.xticks([r + 2*barWidth for r in range(5)], ['L1', 'L2', 'L3', 'L4', 'L5'])
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = OrderedDict(zip(labels, handles))
# plt.legend(
# 	by_label.values(), by_label.keys(), 
# 	ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=16)
# plt.savefig('severity_weather.png')


# ### BLUR
# plt.figure(0) # Here's the part I need

# b1 = [100-(9.0+44.0+12.1+13.9)/4, 100-(9.9+42.6+14.9+14.7)/4, 100-(12.2+41.7+18.6+17.5)/4, 100-(15.3+52.5+19.1+20.5)/4, 68.5]
# b2 = [100-(8.2+40.5+12.2+13.0)/4, 100-(9.1+39.2+16.4+14.2)/4, 100-(12.2+37.9+20.8+17.3)/4, 100-(16.4+50.2+20.7+20.5)/4, 69.0]
# b3 = [100-(8.0+37.9+11.7+12.2)/4, 100-(9.0+36.6+15.4+13.1)/4, 100-(11.5+35.8+19.1+15.8)/4, 100-(15.0+47.8+19.1+18.4)/4, 71.3]
# b4 = [80.8, 80.0, 78.3, 76.1, 74.1]

# # Set position of bar on X axis
# r1 = np.arange(len(b1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
# r4 = [x + barWidth for x in r3]
# r5 = [x + barWidth for x in r4]


# # Make the plot
# plt.bar(r1, b1, color='r', width=barWidth, edgecolor='white', label='B+SS')
# plt.bar(r2, b2, color='y', width=barWidth, edgecolor='white', label='JT')
# plt.bar(r3, b3, color='g', width=barWidth, edgecolor='white', label='TTT')
# plt.bar(r4, b4, color='b', width=barWidth, edgecolor='white', label='Ours')
# # plt.bar(r5, L5, color='#007f5e', width=barWidth, edgecolor='white', label='var5')
 
# # Add xticks on the middle of the group L
# plt.xlabel('Blur', fontweight='bold', fontsize=18)
# plt.xticks([r + 2*barWidth for r in range(5)], ['L1', 'L2', 'L3', 'L4', 'L5'])
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = OrderedDict(zip(labels, handles))
# plt.legend(
# 	by_label.values(), by_label.keys(), 
# 	ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=16)
# plt.savefig('severity_blur.png')


# # # ### NOISE 
plt.figure(0) # Here's the part I need

b1 = [100-(21.7+17.1+17.0)/3, 100-(31.7+22.6+24.3)/3, 100-(42.2+35.1+30.7)/3, 100-(46.4+39.2+44.8)/3, 48.7]
b2 = [100-(20.4+16.6+16.9)/3, 100-(31.0+22.6+23.4)/3, 100-(40.2+34.4+29.9)/3, 100-(45.0+38.3+42.2)/3, 50.6]
b3 = [100-(19.1+15.8+16.5)/3, 100-(28.8+20.7+23.0)/3, 100-(37.2+31.6+28.6)/3, 100-(41.5+35.4+39.8)/3, 54.2]
b4 = [82.0, 77.3, 71.5, 67.2, 65.8]

# Set position of bar on X axis
r1 = np.arange(len(b1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]


# Make the plot
plt.bar(r1, b1, color='r', width=barWidth, edgecolor='white', label='B+SS')
plt.bar(r2, b2, color='y', width=barWidth, edgecolor='white', label='JT')
plt.bar(r3, b3, color='g', width=barWidth, edgecolor='white', label='TTT')
plt.bar(r4, b4, color='b', width=barWidth, edgecolor='white', label='Ours')
# plt.bar(r5, L5, color='#007f5e', width=barWidth, edgecolor='white', label='var5')
 
# Add xticks on the middle of the group L
plt.xlabel('Noise', fontweight='bold', fontsize=18)
plt.xticks([r + barWidth for r in range(5)], ['L1', 'L2', 'L3', 'L4', 'L5'])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(
	by_label.values(), by_label.keys(), 
	ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=16)
plt.savefig('severity_noise.png')






# # # # ### DIGITAL
# plt.figure(0) # Here's the part I need

# b1 = [100-(8.9+9.0+13.2+12.0+17.3)/5, 100-(9.1+10.0+13.1+17.1+22.4)/5, 100-(9.7+11.6+15.3+21.7+24.6)/5, 100-(10.5+13.7+20.8+35.3+26.9)/5, 69.7]
# b2 = [100-(8.1+8.5+12.9+11.3+15.9)/5, 100-(8.3+10.6+12.8+15.9+20.5)/5, 100-(9.2+12.0+15.2+20.8+22.8)/5, 100-(10.0+14.7+19.0+33.2+25.1)/5, 71.6]
# b3 = [100-(8.0+8.3+12.6+11.1+15.5)/5, 100-(8.3+10.2+12.5+14.8+19.7)/5, 100-(9.1+11.6+14.3+18.9+22.3)/5, 100-(10.0+14.1+17.7+29.4+24.5)/5, 73.4]
# b4 = [85.9, 84.4, 82.8, 78.7, 71.6]

# # Set position of bar on X axis
# r1 = np.arange(len(b1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
# r4 = [x + barWidth for x in r3]
# r5 = [x + barWidth for x in r4]


# # Make the plot
# plt.bar(r1, b1, color='r', width=barWidth, edgecolor='white', label='B+SS')
# plt.bar(r2, b2, color='y', width=barWidth, edgecolor='white', label='JT')
# plt.bar(r3, b3, color='g', width=barWidth, edgecolor='white', label='TTT')
# plt.bar(r4, b4, color='b', width=barWidth, edgecolor='white', label='Ours')
# # plt.bar(r5, L5, color='#007f5e', width=barWidth, edgecolor='white', label='var5')
 
# # Add xticks on the middle of the group L
# plt.xlabel('Digital', fontweight='bold', fontsize=18)
# plt.xticks([r + 1.5*barWidth for r in range(5)], ['L1', 'L2', 'L3', 'L4', 'L5'])
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = OrderedDict(zip(labels, handles))
# plt.legend(
# 	by_label.values(), by_label.keys(), 
# 	ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=16)
# plt.savefig('severity_digital.png')



# # ### DEMO

 
# # # # set width of bar
# # # barWidth = 0.25
 
# # # # set height of bar
# # # b1 = [12, 30, 1, 8, 22]
# # # b2 = [28, 6, 16, 5, 10]
# # # b3 = [29, 3, 24, 25, 17]
 
# # # # Set position of bar on X axis
# # # r1 = np.arange(len(b1))
# # # r2 = [x + barWidth for x in r1]
# # # r3 = [x + barWidth for x in r2]
 
# # # # Make the plot
# # # plt.bar(r1, b1, color='#7f6d5f', width=barWidth, edgecolor='white', label='B+SS')
# # # plt.bar(r2, b2, color='#557f2d', width=barWidth, edgecolor='white', label='JT')
# # # plt.bar(r3, b3, color='#2d7f5e', width=barWidth, edgecolor='white', label='TTT')
 
# # # # Add xticks on the middle of the group L
# # # plt.xlabel('group', fontweight='bold')
# # # plt.xticks([r + barWidth for r in range(len(b1))], ['A', 'B', 'C', 'D', 'E'])
 

# # # plt.savefig('severity.png')