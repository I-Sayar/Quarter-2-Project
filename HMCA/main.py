import improve_naive_bayes as inb

def main():
    data, features, table, maxes = inb.get_data()
    inb.complement_naive_bayes(data)
    inb.full_mca(data)
    inb.hierarchical_mca(data, features, table, maxes)

if __name__ == "__main__":
    main()
