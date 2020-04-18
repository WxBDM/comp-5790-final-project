#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:04:43 2020

@author: Brandon
"""
import matplotlib.pyplot as plt

class GraphText:
    
    def __init__(self, ax):
        self.ax = ax
        
        self.title("TITLE GOES HERE.")
        self._y_axis_offset = 0
        self._x_labels_visible = False
    
    def title(self, text):
        bbox_props = dict(boxstyle = "square,pad=0.3", fc = "white", ec = "black", 
                          lw = 1)
        
        self.ax.text(0.5, 1, text, horizontalalignment = 'center', 
                     verticalalignment = 'center', transform = self.ax.transAxes, 
                     fontsize = 'medium', color = 'k', bbox = bbox_props)

    def changeYAxisOffset(self, val):
        self._y_axis_offset = val
    
    def x_axis_label(self, val, **kwargs):
        self.ax.set_xlabel(val, **kwargs)
        self._x_labels_visible = True
    
    def y_axis_label(self, val, **kwargs):
        self.ax.set_ylabel(val, **kwargs)
    
    def show_legend(self):
        if not self._x_labels_visible:
            self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), shadow=True, ncol=3)
        else:
            self.ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, -0.13), shadow=True, ncol=3)

    def upperLeftAbove(self, text):
        self.ax.text(0.01, 1.01, text, horizontalalignment = 'left', verticalalignment = 'bottom',
            transform = self.ax.transAxes, fontsize = 'x-small', color = 'k')
    
    def upperLeftBelow(self, text):
        self.ax.text(0.99, 0.99, text, horizontalalignment = 'left', verticalalignment = 'top',
            transform = self.ax.transAxes, fontsize = 'x-small', color = 'k')
    
    def upperRightAbove(self, text):
        self.ax.text(1, 1.01, text, horizontalalignment = 'right', verticalalignment = 'bottom',
            transform = self.ax.transAxes, fontsize = 'x-small', color = 'k')
    
    def upperRightBelow(self, text):
        self.ax.text(0.99, 0.99, text, horizontalalignment = 'right', verticalalignment = 'top',
            transform = self.ax.transAxes, fontsize = 'x-small', color = 'k')
    
    def save(self, name):
        try:
            plt.savefig(name, dpi=300, format='png', bbox_inches='tight', facecolor = '#F5F5F5')
        except:
            raise ImportError('Be sure to import matplotlib.pyplot as plt.')

if __name__ == "__main__": 
    pass