import platformctrl_pipeline as platformctrl
op = platformctrl.Operator()
a = op.search("含有麻黄的方剂")
print(a[0])
print(a[1])
print(a[2])

x1 = op.load_page(0)
x2 = op.load_page(1)
x3 = op.load_page(2)
