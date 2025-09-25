"""Theme helpers that deliver a polished light/dark appearance."""
from __future__ import annotations

from PyQt6 import QtGui, QtWidgets

LIGHT_QSS = """
QWidget {
    background-color: #f5f5f7;
    color: #1d1d1f;
    font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
    font-size: 14px;
}
QPushButton {
    background-color: #0071e3;
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #0a84ff;
}
QPushButton:disabled {
    background-color: #c5c5c7;
    color: #f0f0f0;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    padding: 6px 8px;
    background: white;
}
QTextEdit, QPlainTextEdit {
    border: 1px solid #d2d2d7;
    border-radius: 12px;
    padding: 12px;
    background: #ffffff;
}
QScrollArea {
    border: none;
}
QLabel.section-title {
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
"""

DARK_QSS = """
QWidget {
    background-color: #1d1d1f;
    color: #f5f5f7;
    font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
    font-size: 14px;
}
QPushButton {
    background-color: #0a84ff;
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #409cff;
}
QPushButton:disabled {
    background-color: #454547;
    color: #8e8e93;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 6px 8px;
    background: #2c2c2e;
    color: #f5f5f7;
}
QTextEdit, QPlainTextEdit {
    border: 1px solid #3a3a3c;
    border-radius: 12px;
    padding: 12px;
    background: #2c2c2e;
    color: #f5f5f7;
}
QScrollArea {
    border: none;
}
QLabel.section-title {
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
"""


def _build_palette(theme: str) -> QtGui.QPalette:
    palette = QtGui.QPalette()

    if theme == "dark":
        base = QtGui.QColor("#1d1d1f")
        alt_base = QtGui.QColor("#2c2c2e")
        text = QtGui.QColor("#f5f5f7")
        tooltip_base = QtGui.QColor("#3a3a3c")
        highlight = QtGui.QColor("#0a84ff")
        highlight_text = QtGui.QColor("#0b0b0d")
        disabled = QtGui.QColor("#5e5e62")
    else:
        base = QtGui.QColor("#f5f5f7")
        alt_base = QtGui.QColor("#ffffff")
        text = QtGui.QColor("#1d1d1f")
        tooltip_base = QtGui.QColor("#ffffff")
        highlight = QtGui.QColor("#0071e3")
        highlight_text = QtGui.QColor("#ffffff")
        disabled = QtGui.QColor("#8e8e93")

    def _apply(color_role: QtGui.QPalette.ColorRole, color: QtGui.QColor) -> None:
        for group in (
            QtGui.QPalette.ColorGroup.Active,
            QtGui.QPalette.ColorGroup.Inactive,
        ):
            palette.setColor(group, color_role, color)

    _apply(QtGui.QPalette.ColorRole.Window, base)
    _apply(QtGui.QPalette.ColorRole.WindowText, text)
    _apply(QtGui.QPalette.ColorRole.Base, alt_base)
    _apply(QtGui.QPalette.ColorRole.AlternateBase, base)
    _apply(QtGui.QPalette.ColorRole.ToolTipBase, tooltip_base)
    _apply(QtGui.QPalette.ColorRole.ToolTipText, text)
    _apply(QtGui.QPalette.ColorRole.Text, text)
    _apply(QtGui.QPalette.ColorRole.Button, alt_base)
    _apply(QtGui.QPalette.ColorRole.ButtonText, text)
    _apply(QtGui.QPalette.ColorRole.Link, highlight)
    _apply(QtGui.QPalette.ColorRole.Highlight, highlight)
    _apply(QtGui.QPalette.ColorRole.HighlightedText, highlight_text)

    palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text, disabled)
    palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ButtonText, disabled)

    return palette


def apply_theme(app: QtWidgets.QApplication, theme: str) -> None:
    palette = _build_palette(theme)
    app.setPalette(palette)
    font = QtGui.QFont("Segoe UI", 11)
    app.setFont(font)
    if theme == "dark":
        app.setStyleSheet(DARK_QSS)
    else:
        app.setStyleSheet(LIGHT_QSS)
