from model import *

if __name__=="__main__":
    # train model and test model
    model=lr_model(num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    
    # draw line of costs
    model_costs=np.squeeze(model["costs"])
    plt.plot(model_costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate = " + str(model["learning_rate"]))
    plt.show()

    # 测试自己的图片
    my_image = "gargouille.jpg"
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(model["num_px"],model["num_px"])).reshape((1, model["num_px"]*model["num_px"]*3)).T
    my_predicted_image = predict(model["w"], model["b"], my_image)
    plt.imshow(image)
    plt.show()
    classes=model["classes"]
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
