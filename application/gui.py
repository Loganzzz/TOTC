from tkinter import ttk
from tkinter import *  
import tkinter.messagebox  
import tkinter.filedialog
from SVMtrain import load, process
from TOCpredict import predict
from logR import logR
import numpy as np
import os
import re
import json

def checkmmap():  
    #清除已删除的model
    if os.path.exists("model"):
        modellist = os.listdir("model")
        if os.path.exists("mmap.json"):
            f = open("mmap.json","r")
            mmap = json.load(f)
            f.close()
            keys = list(mmap.keys())
            for k in keys:
                if k not in modellist:
                    del(mmap[k]) #不能用pop
            f = open("mmap.json","w")
            json.dump(mmap, f)  
            f.close()

class MainWindow:  
  
    def  __init__(self):  
        self.frame = Tk()

        self.frame.title("tight-oil TOC calculator")
        self.frame.minsize(340,440)
        self.frame.maxsize(330,420)

        self.logdata_train = None
        self.tocdata_train = None

        copyright_label = ttk.Label(self.frame,text="copyright@2017张成龙")
        copyright_label.grid(row= 9,column = 0)

        
        def hello():
        	pass
        def about():
        	os.startfile("README.txt")
        #导航菜单
        menubar = Menu(self.frame)
        filemenu = Menu(menubar,tearoff=0)
        filemenu.add_command(label="Open", command=hello)
        filemenu.add_command(label="Save", command=hello)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.frame.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        #创建另一个下拉菜单Edit
        editmenu = Menu(menubar, tearoff=0)
        editmenu.add_command(label="Cut", command=hello)
        editmenu.add_command(label="Copy", command=hello)
        editmenu.add_command(label="Paste", command=hello)
        menubar.add_cascade(label="Edit",menu=editmenu)
        #创建下拉菜单Help
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        #显示菜单
        self.frame.config(menu=menubar)  

        #选项卡
        #tabcontrol
        tabcontrol = ttk.Notebook(self.frame)
        tab1 = ttk.Frame(tabcontrol, width = 250, height = 360)
        tabcontrol.add(tab1, text="模型训练")
        tab2 = ttk.Frame(tabcontrol, width = 250, height = 360)
        tabcontrol.add(tab2, text="TOC预测")
        tab3 = ttk.Frame(tabcontrol, width = 250, height = 360)
        tabcontrol.add(tab3, text="ΔlogR法预测")
        tabcontrol.grid(row=0, column=0)
        #选项卡一中分三部分
        trainframe = ttk.Labelframe(tab1, text="选择训练数据")
        trainframe.grid(row=0, column = 0, padx = 1, pady = 2)
        dataframe = ttk.Labelframe(tab1, text="数据")
        dataframe.grid(row=1, column = 0, padx = 1, pady = 2)
        errorframe = ttk.Labelframe(tab1, text="训练模型")
        errorframe.grid(row=3, column = 0, padx = 1, pady = 2)

        #选项卡二中分三部分
        predictframe = ttk.Labelframe(tab2, text="选择测井数据")
        predictframe.grid(row=0, column = 0, padx = 1, pady = 2)
        modelframe = ttk.Labelframe(tab2, text="模型")
        modelframe.grid(row=1, column = 0, padx = 1, pady = 2)
        fittingframe = ttk.Labelframe(tab2, text="拟合")
        fittingframe.grid(row=4, column = 0, padx = 1, pady = 2)

        #选项卡三中分三部分
        logframe = ttk.Labelframe(tab3, text="选择测井数据")
        logframe.grid(row=0, column = 0, padx = 1, pady = 2)
        baseframe = ttk.Labelframe(tab3, text="基线(可不填，程序自动识别)")
        baseframe.grid(row=1, column = 0, padx = 1, pady = 2)
        calframe = ttk.Labelframe(tab3, text="计算")
        calframe.grid(row=4, column = 0, padx = 1, pady = 2)


        #选项卡一中数据显示
        datacol = ("depth", "AC", "CAL", "CNL", "DEN", "GR", "POR", "RT", "SP")
        showdata = ttk.Treeview(dataframe, columns = datacol,height = 5)
        ysb = ttk.Scrollbar(dataframe, orient='vertical', command=showdata.yview)
        showdata.configure(yscroll=ysb.set)
        showdata.column("depth",width = 10, anchor = "center")
        showdata.column("AC",width = 5, anchor = "center")
        showdata.column("CAL",width = 5, anchor = "center")
        showdata.column("CNL",width = 5, anchor = "center")
        showdata.column("DEN",width = 5, anchor = "center")
        showdata.column("GR",width = 5, anchor = "center")
        showdata.column("POR",width = 5, anchor = "center")
        showdata.column("RT",width = 5, anchor = "center")
        showdata.column("SP",width = 5, anchor = "center")
        showdata.grid(row = 0, column = 1,pady = 1)
        ysb.grid(row = 0,column = 2)

        #选项卡一中选择文件
        label1 = ttk.Label(trainframe,text = "测井数据：")
        label1.grid(row = 1, column = 1, padx = 2)
        log_path = StringVar()
        log_entry = Entry(trainframe,textvariable = log_path)
        log_entry.grid(row=1, column = 2, padx = 2)
        def buttonListener_fileopen(event):
            filename = tkinter.filedialog.askopenfilename()
            if filename!='':
                data = np.loadtxt(filename)
                len = min((np.shape(data)[0], 5))
                for i in range(0, len):
                    showdata.insert("","end",tuple(data[i,:]))
                showdata.update()
                log_path.set(filename)
        button_open = ttk.Button(trainframe,text = "打开文件")
        button_open.grid(row = 1,column = 3, padx = 4)  
        button_open.bind("<ButtonRelease-1>",buttonListener_fileopen)

        label_toc = ttk.Label(trainframe,text = "toc数据：")
        label_toc.grid(row = 2, column = 1, padx = 2)
        toc_path = StringVar()
        toc_entry = Entry(trainframe, textvariable = toc_path)
        toc_entry.grid(row=2, column = 2, padx = 2)
        def buttonListener_tocopen(event):
            filename = tkinter.filedialog.askopenfilename()
            toc_path.set(filename)
        button_tocopen = ttk.Button(trainframe,text = "打开文件")
        button_tocopen.grid(row = 2,column = 3, padx = 4)  
        button_tocopen.bind("<ButtonRelease-1>",buttonListener_tocopen)


        #选项卡一选择测井曲线
        label_log_train = ttk.Labelframe(tab1, text = "")
        label_log_train.grid(row = 2,column = 0)
        tab1_check2_name = StringVar()
        tab1_check2_name.set("AC")
        tab1_check2 = StringVar()
        def tab1_callback2():
        	print(tab1_check2.get())
        tab1_checkbutton2 = ttk.Checkbutton(label_log_train, variable = tab1_check2, textvariable = tab1_check2_name, command = tab1_callback2)
        tab1_checkbutton2.grid(row = 3,column = 1, padx = 10)

        tab1_check3_name = StringVar()
        tab1_check3_name.set("CAL")
        tab1_check3 = StringVar()
        def tab1_callback3():
        	print(tab1_check3.get())
        tab1_checkbutton3 = ttk.Checkbutton(label_log_train, variable = tab1_check3, textvariable = tab1_check3_name, command = tab1_callback3)
        tab1_checkbutton3.grid(row = 3,column = 2, padx = 10)

        tab1_check4_name = StringVar()
        tab1_check4_name.set("CNL")
        tab1_check4 = StringVar()
        def tab1_callback4():
        	print(tab1_check4.get())
        tab1_checkbutton4 = ttk.Checkbutton(label_log_train, variable = tab1_check4, textvariable = tab1_check4_name, command = tab1_callback4)
        tab1_checkbutton4.grid(row = 3,column = 3, padx = 10)

        tab1_check5_name = StringVar()
        tab1_check5_name.set("DEN")
        tab1_check5 = StringVar()
        def tab1_callback5():
        	print(tab1_check5.get())
        tab1_checkbutton5 = ttk.Checkbutton(label_log_train, variable = tab1_check5, textvariable = tab1_check5_name, command = tab1_callback5)
        tab1_checkbutton5.grid(row = 3,column = 4, padx = 10)

        tab1_check6_name = StringVar()
        tab1_check6_name.set("GR")
        tab1_check6 = StringVar()
        def tab1_callback6():
        	print(tab1_check6.get())
        tab1_checkbutton6 = ttk.Checkbutton(label_log_train, variable = tab1_check6, textvariable = tab1_check6_name, command = tab1_callback6)
        tab1_checkbutton6.grid(row = 4,column = 1, padx = 10)

        tab1_check7_name = StringVar()
        tab1_check7_name.set("POR")
        tab1_check7 = StringVar()
        def tab1_callback7():
        	print(tab1_check7.get())
        tab1_checkbutton7 = ttk.Checkbutton(label_log_train, variable = tab1_check7, textvariable = tab1_check7_name, command = tab1_callback7)
        tab1_checkbutton7.grid(row = 4,column = 2, padx = 10)       

        tab1_check8_name = StringVar()
        tab1_check8_name.set("RT ")
        tab1_check8 = StringVar()
        def tab1_callback8():
        	print(tab1_check8.get())
        tab1_checkbutton8 = ttk.Checkbutton(label_log_train, variable = tab1_check8, textvariable = tab1_check8_name, command = tab1_callback8)
        tab1_checkbutton8.grid(row = 4,column = 3, padx = 10)

        tab1_check9_name = StringVar()
        tab1_check9_name.set("SP ")
        tab1_check9 = StringVar()
        def tab1_callback9():
        	print(tab1_check9.get())
        tab1_checkbutton9 = ttk.Checkbutton(label_log_train, variable = tab1_check9, textvariable = tab1_check9_name, command = tab1_callback9)
        tab1_checkbutton9.grid(row = 4,column = 4, padx = 10)

 		#选项卡一中复选键
        tab1_check_es_name = StringVar()
        tab1_check_es_name.set("误差累计曲线")
        tab1_check_es = StringVar()
        tab1_checkbutton_es = ttk.Checkbutton(errorframe, variable = tab1_check_es, textvariable = tab1_check_es_name)
        tab1_checkbutton_es.grid(row = 0,column = 0,pady = 2)

        tab1_check_pr_name = StringVar()
        tab1_check_pr_name.set("predict-real")
        tab1_check_pr = StringVar()
        tab1_checkbutton_pr = ttk.Checkbutton(errorframe, variable = tab1_check_pr, textvariable = tab1_check_pr_name)
        tab1_checkbutton_pr.grid(row = 0,column = 1,pady = 2)

        tab1_check_line_name = StringVar()
        tab1_check_line_name.set("predict-line")
        tab1_check_line = StringVar()
        tab1_checkbutton_line = ttk.Checkbutton(errorframe, variable = tab1_check_line, textvariable = tab1_check_line_name)
        tab1_checkbutton_line.grid(row = 0,column = 2,pady = 2)

        #选项卡一中绘图
        #此处调用模型训练函数
        def buttonListener_plot_error(event):
            
            
            #检查toc文件和测井文件
            if toc_path.get() != "":
                self.tocdata_train = load(toc_path.get())
            else:
                tkinter.messagebox.showinfo("messagebox","请输入TOC数据文件")
                return 
            if log_path.get() != "":
                self.logdata_train = load(log_path.get())
            else:
                tkinter.messagebox.showinfo("messagebox","请输入测井数据文件")
                return 
            
            #检查要画的曲线
            plot_pr = 0 
            plot_es = 0
            plot_line = 0
            if tab1_check_pr.get() == "1":
                plot_pr = 1
            if tab1_check_es.get() == "1":
                plot_es = 1
            if tab1_check_line.get() == "1":
                plot_line = 1 
            
            #检查要用到的测井曲线
            tab1_check_list_log = []
            loglist = []
            if tab1_check2.get() == "1":
                tab1_check_list_log.append(1)
                loglist.append("AC")
            if tab1_check3.get() == "1":
                tab1_check_list_log.append(2)
                loglist.append("CAL")
            if tab1_check4.get() == "1":
                tab1_check_list_log.append(3)
                loglist.append("CNL")
            if tab1_check5.get() == "1":
                tab1_check_list_log.append(4)
                loglist.append("DEN")
            if tab1_check6.get() == "1":
                tab1_check_list_log.append(5)
                loglist.append("GR")
            if tab1_check7.get() == "1":
                tab1_check_list_log.append(6)
                loglist.append("POR")
            if tab1_check8.get() == "1":
                tab1_check_list_log.append(7)
                loglist.append("RT")
            if tab1_check9.get() == "1":
                tab1_check_list_log.append(8)
                loglist.append("SP")
            if len(tab1_check_list_log) == 0:
                tkinter.messagebox.showinfo("messagebox","测井曲线不能为空")
            else: 
                tkinter.messagebox.showinfo("messagebox","训练需要1-3min，请耐心等待,放松一下http://www.4399.com/flash/7361.htm#search3\nhttp://www.u77.com/game/6015")
                process(self.logdata_train,self.tocdata_train,tab1_check_list_log,plot_pr,plot_es,plot_line,loglist)
                tkinter.messagebox.showinfo("messagebox","完成")
                
        button_open = ttk.Button(errorframe,text = "绘制曲线")
        button_open.grid(row = 1,column = 1, padx = 4,pady = 2)  
        button_open.bind("<ButtonRelease-1>",buttonListener_plot_error)

        #选项卡二

        #选项卡二中选择文件
        label_logopen = ttk.Label(predictframe,text = "测井数据：")
        label_logopen.grid(row = 1, column = 1, padx = 2)
        log_path_pre = StringVar()
        log_entry_pre = ttk.Entry(predictframe,textvariable = log_path_pre)
        log_entry_pre.grid(row=1, column = 2, padx = 2)
        def buttonListener_logopen(event):
            filename = tkinter.filedialog.askopenfilename()
            log_path_pre.set(filename)
        button_logopen = ttk.Button(predictframe,text = "打开文件")
        button_logopen.grid(row = 1,column = 3, padx = 4)  
        button_logopen.bind("<ButtonRelease-1>",buttonListener_logopen)
        
         #选项卡二选择测井曲线
        label_log_select = ttk.Labelframe(tab2, text = "使用的测井曲线")
        label_log_select.grid(row = 2,column = 0)
        tab2_check2_name = StringVar()
        tab2_check2_name.set("AC")
        tab2_check2 = StringVar()
        tab2_checkbutton2 = Checkbutton(label_log_select, variable = tab2_check2, textvariable = tab2_check2_name, stat="disabled")
        tab2_checkbutton2.grid(row = 3,column = 1, padx = 10)

        tab2_check3_name = StringVar()
        tab2_check3_name.set("CAL")
        tab2_check3 = StringVar()
        tab2_checkbutton3 = Checkbutton(label_log_select, variable = tab2_check3, textvariable = tab2_check3_name, stat="disabled")
        tab2_checkbutton3.grid(row = 3,column = 2, padx = 10)

        tab2_check4_name = StringVar()
        tab2_check4_name.set("CNL")
        tab2_check4 = StringVar()
        tab2_checkbutton4 = Checkbutton(label_log_select, variable = tab2_check4, textvariable = tab2_check4_name, stat="disabled")
        tab2_checkbutton4.grid(row = 3,column = 3, padx = 10)

        tab2_check5_name = StringVar()
        tab2_check5_name.set("DEN")
        tab2_check5 = StringVar()
        tab2_checkbutton5 = Checkbutton(label_log_select, variable = tab2_check5, textvariable = tab2_check5_name, stat="disabled")
        tab2_checkbutton5.grid(row = 3,column = 4, padx = 10)

        tab2_check6_name = StringVar()
        tab2_check6_name.set("GR")
        tab2_check6 = StringVar()
        tab2_checkbutton6 = Checkbutton(label_log_select, variable = tab2_check6, textvariable = tab2_check6_name, stat="disabled")
        tab2_checkbutton6.grid(row = 4,column = 1, padx = 10)

        tab2_check7_name = StringVar()
        tab2_check7_name.set("POR")
        tab2_check7 = StringVar()
        tab2_checkbutton7 = Checkbutton(label_log_select, variable = tab2_check7, textvariable = tab2_check7_name, stat="disabled")
        tab2_checkbutton7.grid(row = 4,column = 2, padx = 10)       

        tab2_check8_name = StringVar()
        tab2_check8_name.set("RT ")
        tab2_check8 = StringVar()
        tab2_checkbutton8 = Checkbutton(label_log_select, variable = tab2_check8, textvariable = tab2_check8_name, stat="disabled")
        tab2_checkbutton8.grid(row = 4,column = 3, padx = 10)

        tab2_check9_name = StringVar()
        tab2_check9_name.set("SP ")
        tab2_check9 = StringVar()
        tab2_checkbutton9 = Checkbutton(label_log_select, variable = tab2_check9, textvariable = tab2_check9_name, stat="disabled")
        tab2_checkbutton9.grid(row = 4,column = 4, padx = 10)      


        #选项卡二选择模型
        label_model_select = ttk.Label(modelframe,text = "选择模型：")
        label_model_select.grid(row = 2, column = 1, padx = 2)

        v = StringVar(modelframe)
        cb_model = ttk.Combobox(modelframe,textvariable = v)
        def getmodels():
            model_list = []  #获取文件目录下的所有模型
            if os.path.exists("model"):
                filelist = os.listdir("model")
                modelpattern = re.compile(r'.*?pkl', re.I)
                for line in filelist:
                    if modelpattern.match(line):
                        model_list.append(line)
            if len(model_list) == 0:
                v.set("none")
            return model_list
        cb_model["value"] = getmodels()       
        def updatemodel(event):       #更新模型数量的事件，为了方便
            cb_model["value"] = getmodels()
            currentmodel = v.get()
            #关联测井曲线
            if os.path.exists("mmap.json"):
                f = open("mmap.json")
                mmap = json.load(f)
                if currentmodel in mmap.keys():  
                    logused = mmap[currentmodel]
                    if "AC" in logused:
                        tab2_checkbutton2.select()
                    else:
                        tab2_checkbutton2.deselect()
                    if "CAL" in logused:
                        tab2_checkbutton3.select()
                    else:
                        tab2_checkbutton3.deselect()
                    if "CNL" in logused:
                        tab2_checkbutton4.select()
                    else:
                        tab2_checkbutton4.deselect()
                    if "DEN" in logused:
                        tab2_checkbutton5.select()
                    else:
                        tab2_checkbutton5.deselect()
                    if "GR" in logused:
                        tab2_checkbutton6.select()
                    else:
                        tab2_checkbutton6.deselect()
                    if "POR" in logused:
                        tab2_checkbutton7.select()
                    else:
                        tab2_checkbutton7.deselect()
                    if "RT" in logused:
                        tab2_checkbutton8.select()
                    else:
                        tab2_checkbutton8.deselect()
                    if "SP" in logused:
                        tab2_checkbutton9.select()
                    else:
                        tab2_checkbutton9.deselect()

                
        cb_model.bind("<<ComboboxSelected>>", updatemodel)         
        cb_model.grid(row = 2,column = 2,padx = 15)

        #选项卡二 复选键
        check1_name = StringVar()
        check1_name.set("绘制曲线")
        check1 = StringVar()
        checkbutton1 = ttk.Checkbutton(fittingframe, variable = check1, textvariable = check1_name)
        checkbutton1.grid(row = 1,column = 2)

       
        #选项卡二 拟合按钮
        def buttonListener_fit(event):
            log_array = []
            
            if log_path_pre.get() == "": #没有测井文件
                tkinter.messagebox.showinfo("提示","请输入测井文件")
            else:
                if not os.path.exists("mmap.json"):
                    tkinter.messagebox.showinfo("警告","缺少模型映射文件mmap.json")
                else:
                    currentmodel = v.get()
                    #关联测井曲线
                    f = open("mmap.json")
                    mmap = json.load(f)
                    if currentmodel not in mmap.keys(): #映射文件中没有当前模型
                        tkinter.messagebox.showinfo("警告","当前模型不可用")
                    else:
                        filename = tkinter.filedialog.asksaveasfilename(filetypes=(("Text","*.txt"),("All files","*.*")))
                        if not filename == "":        #取消了
                            logused = mmap[currentmodel]
                            print(logused)
                            if "AC" in logused:
                                log_array.append(1)
                            if "CAL" in logused:
                                log_array.append(2)
                            if "CNL" in logused:
                                log_array.append(3)
                            if "DEN" in logused:
                                log_array.append(4)
                            if "GR" in logused:
                                log_array.append(5)
                            if "POR" in logused:
                                log_array.append(6)
                            if "RT" in logused:
                                log_array.append(7)
                            if "SP" in logused:
                                log_array.append(8)
                            log_data_pre = load(log_path_pre.get())
                            plot_pre = 0
                            if check1.get() == "1":   #是否勾选画曲线
                                plot_pre = 1
                                
                            predict("model/" +currentmodel, log_data_pre, log_array, plot_pre, filename+".txt")
                            tkinter.messagebox.showinfo("提示","预测完成")
                            
        button_fit = ttk.Button(fittingframe,text = "拟合")
        button_fit.grid(row = 3,column = 2, padx = 4)  
        button_fit.bind("<ButtonRelease-1>",buttonListener_fit)
        
        #选项卡三中选择文件
        label_dlog = ttk.Label(logframe,text = "测井数据：")
        label_dlog.grid(row = 1, column = 1, padx = 2)
        dlog_path = StringVar()
        dlog_entry = Entry(logframe,textvariable = dlog_path)
        dlog_entry.grid(row=1, column = 2, padx = 2)
        def buttonListener_dlogopen(event):
            filename = tkinter.filedialog.askopenfilename()
            dlog_path.set(filename)
        button_dlog = ttk.Button(logframe,text = "打开文件")
        button_dlog.grid(row = 1,column = 3, padx = 4)  
        button_dlog.bind("<ButtonRelease-1>",buttonListener_dlogopen)
        
        #选项卡三中选择基线
        label_abase = ttk.Label(baseframe,text = "声波时差基线：")
        label_abase.grid(row = 1, column = 1, padx = 2)
        abase = StringVar()
        abase_entry = Entry(baseframe,textvariable = abase)
        abase_entry.grid(row=1, column = 2, padx = 2)
        label_abase_unit = ttk.Label(baseframe,text = "(μs/ft)")
        label_abase_unit.grid(row = 1, column = 3, padx = 2)
        
        label_rbase = ttk.Label(baseframe,text = "电阻率基线：")
        label_rbase.grid(row = 2, column = 1, padx = 2)
        rbase = StringVar()
        rbase_entry = Entry(baseframe,textvariable = rbase)
        rbase_entry.grid(row=2, column = 2, padx = 2)
        label_rbase_unit = ttk.Label(baseframe,text = "(omm)")
        label_rbase_unit.grid(row = 2, column = 3, padx = 2)
        
        label_tbase = ttk.Label(baseframe,text = "TOC基线(默认0)")
        label_tbase.grid(row = 3, column = 1, padx = 2)
        tbase = StringVar()
        tbase_entry = Entry(baseframe,textvariable = tbase)
        tbase_entry.grid(row=3, column = 2, padx = 2)
        label_tbase_unit = ttk.Label(baseframe,text = "(%)")
        label_tbase_unit.grid(row = 3, column = 3, padx = 2)

        label_lbase = ttk.Label(baseframe,text = "热变指数lom(默认15)")
        label_lbase.grid(row = 4, column = 1, padx = 2)
        lbase = StringVar()
        lbase_entry = Entry(baseframe,textvariable = lbase)
        lbase_entry.grid(row=4, column = 2, padx = 2)
        label_lbase_unit = ttk.Label(baseframe,text = "(-)")
        label_lbase_unit.grid(row = 4, column = 3, padx = 2)
        
         #选项卡三 预测按钮
        def buttonListener_logfit(event):
            if dlog_path.get() == "": #没有测井文件
                tkinter.messagebox.showinfo("提示","请输入测井文件")
            else:
                logr_data = dlog_path.get()
                filename_logr = tkinter.filedialog.asksaveasfilename(filetypes=(("Text","*.txt"),("All files","*.*")))
                if not filename_logr == "":        #取消了
                    lom = lbase.get()
                    base_toc= tbase.get()
                    if lom=='':
                        lom = 15
                    if base_toc == '':
                        base_toc = 0
                        
                    final_ac,final_rt = logR(filename_logr+'.txt',logr_data, base_ac=abase.get(), base_rt=rbase.get(), k=0.02, lom=lom, base_toc=base_toc)
                    tkinter.messagebox.showinfo("提示",("预测完成\nAC基线：%s;RT基线:%s"%(final_ac,final_rt)))
                            
        button_logfit = ttk.Button(calframe,text = "预测")
        button_logfit.grid(row = 1,column = 1, padx = 4)  
        button_logfit.bind("<ButtonRelease-1>",buttonListener_logfit)
        
       

        self.frame.mainloop()  
  