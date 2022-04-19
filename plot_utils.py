import matplotlib.pyplot as plt

def bar_chart(x,y,title):
    plt.figure()
    plt.bar(x,y)
    plt.title(title)
    plt.xticks(rotation = 45) 
    plt.show

def pie_chart(x, y, title):
    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.title(title)
    plt.show()

def histogram(data, title):
    plt.figure()
    plt.hist(data) # default is 10
    plt.xticks(rotation = 45)
    plt.title(title)
    plt.show()

def linear_regression(x, y):
    plt.figure() 
    plt.scatter(x, y)
    plt.margins(x=0, y=0)
    plt.show()

def box_plot(distributions): # distributions and labels are parallel
    # distributions: list of 1D lists of values
    plt.figure()
    plt.boxplot(distributions)
    
    # plt.xticks(list(range(1, len(distributions) + 1)), labels)
    plt.show()