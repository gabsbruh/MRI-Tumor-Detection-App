from app.MainWindow import MainWindow

def test_navigation(qtbot):
    main_window = MainWindow()
    qtbot.addWidget(main_window)
    assert main_window.stacked_widget.currentWidget() == main_window.start_page
    main_window.show_page("SinglePage")
    assert main_window.stacked_widget.currentWidget() == main_window.single_page