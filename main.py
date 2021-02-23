import argparse
import img_processing as Ip

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("path",type=str,help="image path")
    parser.add_argument("-m","--mode",type=str,required=True,help="Mode => \"effects\" (add filters), \"replace\" (replace faces)")
    parser.add_argument("-t","--type",help="Type in effects mode => \n blur,cartoon,flip,xi,beauty,pewdiepie")
    parser.add_argument("-o",help="oultine around box in replace mode",action="store_true")
    args=parser.parse_args()

    if(args.mode=="effects"):
        if(args.type==None):
            print("Type missing!")
        else:    
            Ip.process(args.path).effects(args.type)
    elif(args.mode=="replace"):
        BOX=False if args.type=="noBox" else True
        Ip.process(args.path).boxFace(BOX=BOX,OUTLINE=args.o)

    else:
        print("Invalid arguments")
        



