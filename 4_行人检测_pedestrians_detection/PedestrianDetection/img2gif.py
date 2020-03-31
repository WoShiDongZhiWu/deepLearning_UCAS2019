'''将img转换为gif图'''
import imageio 

def compose_gif(): 
    img_paths = []
    for i in range(4500,5056):
        img_paths.append('Dataset/result_test_images/%a.jpg.jpg'%(i))
    gif_images = [] 
    for path in img_paths: 
        gif_images.append(imageio.imread(path)) 
    imageio.mimsave("result.gif",gif_images,fps=20)

if __name__ == "__main__":
    compose_gif()