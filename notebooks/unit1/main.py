from pyvirtualdisplay import Display


def main():
    # Virtual display
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()


if __name__ == "__main__":
    main()
