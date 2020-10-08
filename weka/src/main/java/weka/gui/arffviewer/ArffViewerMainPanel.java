/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * ArffViewerMainPanel.java
 * Copyright (C) 2005-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.gui.arffviewer;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Cursor;
import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Vector;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JInternalFrame;
import javax.swing.JList;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.KeyStroke;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.converters.AbstractFileLoader;
import weka.core.converters.AbstractFileSaver;
import weka.core.converters.AbstractSaver;
import weka.core.converters.ConverterUtils;
import weka.gui.ComponentHelper;
import weka.gui.ConverterFileChooser;
import weka.gui.JTableHelper;
import weka.gui.ListSelectorDialog;

/**
 * The main panel of the ArffViewer. It has a reference to the menu, that an
 * implementing JFrame only needs to add via the setJMenuBar(JMenuBar) method.
 *
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */

public class ArffViewerMainPanel extends JPanel implements ActionListener, ChangeListener {

	/** for serialization */
	static final long serialVersionUID = -8763161167586738753L;

	/** the default for width */
	public final static int DEFAULT_WIDTH = -1;
	/** the default for height */
	public final static int DEFAULT_HEIGHT = -1;
	/** the default for left */
	public final static int DEFAULT_LEFT = -1;
	/** the default for top */
	public final static int DEFAULT_TOP = -1;
	/** default width */
	public final static int WIDTH = 800;
	/** default height */
	public final static int HEIGHT = 600;

	protected Container parent;
	protected JTabbedPane tabbedPane;
	protected JMenuBar menuBar;
	protected JMenu menuFile;
	protected JMenuItem menuFileOpen;
	protected JMenuItem menuFileSave;
	protected JMenuItem menuFileSaveAs;
	protected JMenuItem menuFileClose;
	protected JMenuItem menuFileCloseAll;
	protected JMenuItem menuFileProperties;
	protected JMenuItem menuFileExit;
	protected JMenu menuEdit;
	protected JMenuItem menuEditUndo;
	protected JMenuItem menuEditCopy;
	protected JMenuItem menuEditSearch;
	protected JMenuItem menuEditClearSearch;
	protected JMenuItem menuEditDeleteAttribute;
	protected JMenuItem menuEditDeleteAttributes;
	protected JMenuItem menuEditRenameAttribute;
	protected JMenuItem menuEditAttributeAsClass;
	protected JMenuItem menuEditDeleteInstance;
	protected JMenuItem menuEditDeleteInstances;
	protected JMenuItem menuEditSortInstances;
	protected JMenu menuView;
	protected JMenuItem menuViewAttributes;
	protected JMenuItem menuViewValues;
	protected JMenuItem menuViewOptimalColWidths;

	protected ConverterFileChooser fileChooser;
	protected String frameTitle;
	protected boolean confirmExit;
	protected int width;
	protected int height;
	protected int top;
	protected int left;
	protected boolean exitOnClose;

	/**
	 * initializes the object
	 *
	 * @param parentFrame the parent frame (JFrame or JInternalFrame)
	 */
	public ArffViewerMainPanel(final Container parentFrame) {
		this.parent = parentFrame;
		this.frameTitle = "ARFF-Viewer";
		this.createPanel();
	}

	/**
	 * creates all the components in the panel
	 */
	protected void createPanel() {
		// basic setup
		this.setSize(WIDTH, HEIGHT);

		this.setConfirmExit(false);
		this.setLayout(new BorderLayout());

		// file dialog
		this.fileChooser = new ConverterFileChooser(new File(System.getProperty("user.dir")));
		this.fileChooser.setMultiSelectionEnabled(true);

		// menu
		this.menuBar = new JMenuBar();
		this.menuFile = new JMenu("File");
		this.menuFileOpen = new JMenuItem("Open...", ComponentHelper.getImageIcon("open.gif"));
		this.menuFileOpen.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_O, KeyEvent.CTRL_MASK));
		this.menuFileOpen.addActionListener(this);
		this.menuFileSave = new JMenuItem("Save", ComponentHelper.getImageIcon("save.gif"));
		this.menuFileSave.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_S, KeyEvent.CTRL_MASK));
		this.menuFileSave.addActionListener(this);
		this.menuFileSaveAs = new JMenuItem("Save as...", ComponentHelper.getImageIcon("empty.gif"));
		this.menuFileSaveAs.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_S, KeyEvent.CTRL_MASK + KeyEvent.SHIFT_MASK));
		this.menuFileSaveAs.addActionListener(this);
		this.menuFileClose = new JMenuItem("Close", ComponentHelper.getImageIcon("empty.gif"));
		this.menuFileClose.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_W, KeyEvent.CTRL_MASK));
		this.menuFileClose.addActionListener(this);
		this.menuFileCloseAll = new JMenuItem("Close all", ComponentHelper.getImageIcon("empty.gif"));
		this.menuFileCloseAll.addActionListener(this);
		this.menuFileProperties = new JMenuItem("Properties", ComponentHelper.getImageIcon("empty.gif"));
		this.menuFileProperties.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_ENTER, KeyEvent.CTRL_MASK));
		this.menuFileProperties.addActionListener(this);
		this.menuFileExit = new JMenuItem("Exit", ComponentHelper.getImageIcon("forward.gif"));
		this.menuFileExit.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_X, KeyEvent.ALT_MASK));
		this.menuFileExit.addActionListener(this);
		this.menuFile.add(this.menuFileOpen);
		this.menuFile.add(this.menuFileSave);
		this.menuFile.add(this.menuFileSaveAs);
		this.menuFile.add(this.menuFileClose);
		this.menuFile.add(this.menuFileCloseAll);
		this.menuFile.addSeparator();
		this.menuFile.add(this.menuFileProperties);
		this.menuFile.addSeparator();
		this.menuFile.add(this.menuFileExit);
		this.menuBar.add(this.menuFile);

		this.menuEdit = new JMenu("Edit");
		this.menuEditUndo = new JMenuItem("Undo", ComponentHelper.getImageIcon("undo.gif"));
		this.menuEditUndo.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_Z, KeyEvent.CTRL_MASK));
		this.menuEditUndo.addActionListener(this);
		this.menuEditCopy = new JMenuItem("Copy", ComponentHelper.getImageIcon("copy.gif"));
		this.menuEditCopy.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_INSERT, KeyEvent.CTRL_MASK));
		this.menuEditCopy.addActionListener(this);
		this.menuEditSearch = new JMenuItem("Search...", ComponentHelper.getImageIcon("find.gif"));
		this.menuEditSearch.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F, KeyEvent.CTRL_MASK));
		this.menuEditSearch.addActionListener(this);
		this.menuEditClearSearch = new JMenuItem("Clear search", ComponentHelper.getImageIcon("empty.gif"));
		this.menuEditClearSearch.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F, KeyEvent.CTRL_MASK + KeyEvent.SHIFT_MASK));
		this.menuEditClearSearch.addActionListener(this);
		this.menuEditRenameAttribute = new JMenuItem("Rename attribute", ComponentHelper.getImageIcon("empty.gif"));
		this.menuEditRenameAttribute.addActionListener(this);
		this.menuEditAttributeAsClass = new JMenuItem("Attribute as class", ComponentHelper.getImageIcon("empty.gif"));
		this.menuEditAttributeAsClass.addActionListener(this);
		this.menuEditDeleteAttribute = new JMenuItem("Delete attribute", ComponentHelper.getImageIcon("empty.gif"));
		this.menuEditDeleteAttribute.addActionListener(this);
		this.menuEditDeleteAttributes = new JMenuItem("Delete attributes", ComponentHelper.getImageIcon("empty.gif"));
		this.menuEditDeleteAttributes.addActionListener(this);
		this.menuEditDeleteInstance = new JMenuItem("Delete instance", ComponentHelper.getImageIcon("empty.gif"));
		this.menuEditDeleteInstance.addActionListener(this);
		this.menuEditDeleteInstances = new JMenuItem("Delete instances", ComponentHelper.getImageIcon("empty.gif"));
		this.menuEditDeleteInstances.addActionListener(this);
		this.menuEditSortInstances = new JMenuItem("Sort data (ascending)", ComponentHelper.getImageIcon("sort.gif"));
		this.menuEditSortInstances.addActionListener(this);
		this.menuEdit.add(this.menuEditUndo);
		this.menuEdit.addSeparator();
		this.menuEdit.add(this.menuEditCopy);
		this.menuEdit.addSeparator();
		this.menuEdit.add(this.menuEditSearch);
		this.menuEdit.add(this.menuEditClearSearch);
		this.menuEdit.addSeparator();
		this.menuEdit.add(this.menuEditRenameAttribute);
		this.menuEdit.add(this.menuEditAttributeAsClass);
		this.menuEdit.add(this.menuEditDeleteAttribute);
		this.menuEdit.add(this.menuEditDeleteAttributes);
		this.menuEdit.addSeparator();
		this.menuEdit.add(this.menuEditDeleteInstance);
		this.menuEdit.add(this.menuEditDeleteInstances);
		this.menuEdit.add(this.menuEditSortInstances);
		this.menuBar.add(this.menuEdit);

		this.menuView = new JMenu("View");
		this.menuViewAttributes = new JMenuItem("Attributes...", ComponentHelper.getImageIcon("objects.gif"));
		this.menuViewAttributes.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_A, KeyEvent.CTRL_MASK + KeyEvent.SHIFT_MASK));
		this.menuViewAttributes.addActionListener(this);
		this.menuViewValues = new JMenuItem("Values...", ComponentHelper.getImageIcon("properties.gif"));
		this.menuViewValues.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_V, KeyEvent.CTRL_MASK + KeyEvent.SHIFT_MASK));
		this.menuViewValues.addActionListener(this);
		this.menuViewOptimalColWidths = new JMenuItem("Optimal column width (all)", ComponentHelper.getImageIcon("resize.gif"));
		this.menuViewOptimalColWidths.addActionListener(this);
		this.menuView.add(this.menuViewAttributes);
		this.menuView.add(this.menuViewValues);
		this.menuView.addSeparator();
		this.menuView.add(this.menuViewOptimalColWidths);
		this.menuBar.add(this.menuView);

		// tabbed pane
		this.tabbedPane = new JTabbedPane();
		this.tabbedPane.addChangeListener(this);
		this.add(this.tabbedPane, BorderLayout.CENTER);

		this.updateMenu();
		this.updateFrameTitle();
	}

	/**
	 * returns the parent frame, if it's a JFrame, otherwise null
	 *
	 * @return the parent frame
	 */
	public JFrame getParentFrame() {
		if (this.parent instanceof JFrame) {
			return (JFrame) this.parent;
		} else {
			return null;
		}
	}

	/**
	 * returns the parent frame, if it's a JInternalFrame, otherwise null
	 *
	 * @return the parent frame
	 */
	public JInternalFrame getParentInternalFrame() {
		if (this.parent instanceof JInternalFrame) {
			return (JInternalFrame) this.parent;
		} else {
			return null;
		}
	}

	/**
	 * sets the new parent frame
	 *
	 * @param value the parent frame
	 */
	public void setParent(final Container value) {
		this.parent = value;
	}

	/**
	 * returns the menu bar to be added in a frame
	 *
	 * @return the menu bar
	 */
	public JMenuBar getMenu() {
		return this.menuBar;
	}

	/**
	 * returns the tabbedpane instance
	 *
	 * @return the tabbed pane
	 */
	public JTabbedPane getTabbedPane() {
		return this.tabbedPane;
	}

	/**
	 * whether to present a MessageBox on Exit or not
	 *
	 * @param confirm whether a MessageBox pops up or not to confirm exit
	 */
	public void setConfirmExit(final boolean confirm) {
		this.confirmExit = confirm;
	}

	/**
	 * returns the setting of whether to display a confirm messagebox or not on
	 * exit
	 *
	 * @return whether a messagebox is displayed or not
	 */
	public boolean getConfirmExit() {
		return this.confirmExit;
	}

	/**
	 * whether to do a System.exit(0) on close
	 *
	 * @param value enables/disables a System.exit(0) on close
	 */
	public void setExitOnClose(final boolean value) {
		this.exitOnClose = value;
	}

	/**
	 * returns TRUE if a System.exit(0) is done on a close
	 *
	 * @return true if a System.exit(0) is done on close
	 */
	public boolean getExitOnClose() {
		return this.exitOnClose;
	}

	/**
	 * validates and repaints the frame
	 */
	public void refresh() {
		this.validate();
		this.repaint();
	}

	/**
	 * returns the title (incl. filename) for the frame
	 *
	 * @return the frame title
	 */
	public String getFrameTitle() {
		if (this.getCurrentFilename().equals("")) {
			return this.frameTitle;
		} else {
			return this.frameTitle + " - " + this.getCurrentFilename();
		}
	}

	/**
	 * sets the title of the parent frame, if one was provided
	 */
	public void updateFrameTitle() {
		if (this.getParentFrame() != null) {
			this.getParentFrame().setTitle(this.getFrameTitle());
		}
		if (this.getParentInternalFrame() != null) {
			this.getParentInternalFrame().setTitle(this.getFrameTitle());
		}
	}

	/**
	 * sets the enabled/disabled state of the menu
	 */
	protected void updateMenu() {
		boolean fileOpen;
		boolean isChanged;
		boolean canUndo;

		fileOpen = (this.getCurrentPanel() != null);
		isChanged = fileOpen && (this.getCurrentPanel().isChanged());
		canUndo = fileOpen && (this.getCurrentPanel().canUndo());

		// File
		this.menuFileOpen.setEnabled(true);
		this.menuFileSave.setEnabled(isChanged);
		this.menuFileSaveAs.setEnabled(fileOpen);
		this.menuFileClose.setEnabled(fileOpen);
		this.menuFileCloseAll.setEnabled(fileOpen);
		this.menuFileProperties.setEnabled(fileOpen);
		this.menuFileExit.setEnabled(true);
		// Edit
		this.menuEditUndo.setEnabled(canUndo);
		this.menuEditCopy.setEnabled(fileOpen);
		this.menuEditSearch.setEnabled(fileOpen);
		this.menuEditClearSearch.setEnabled(fileOpen);
		this.menuEditAttributeAsClass.setEnabled(fileOpen);
		this.menuEditRenameAttribute.setEnabled(fileOpen);
		this.menuEditDeleteAttribute.setEnabled(fileOpen);
		this.menuEditDeleteAttributes.setEnabled(fileOpen);
		this.menuEditDeleteInstance.setEnabled(fileOpen);
		this.menuEditDeleteInstances.setEnabled(fileOpen);
		this.menuEditSortInstances.setEnabled(fileOpen);
		// View
		this.menuViewAttributes.setEnabled(fileOpen);
		this.menuViewValues.setEnabled(fileOpen);
		this.menuViewOptimalColWidths.setEnabled(fileOpen);
	}

	/**
	 * sets the title of the tab that contains the given component
	 *
	 * @param component the component to set the title for
	 */
	protected void setTabTitle(final JComponent component) {
		int index;

		if (!(component instanceof ArffPanel)) {
			return;
		}

		index = this.tabbedPane.indexOfComponent(component);
		if (index == -1) {
			return;
		}

		this.tabbedPane.setTitleAt(index, ((ArffPanel) component).getTitle());
		this.updateFrameTitle();
	}

	/**
	 * returns the number of panels currently open
	 *
	 * @return the number of open panels
	 */
	public int getPanelCount() {
		return this.tabbedPane.getTabCount();
	}

	/**
	 * returns the specified panel, <code>null</code> if index is out of bounds
	 *
	 * @param index the index of the panel
	 * @return the panel
	 */
	public ArffPanel getPanel(final int index) {
		if ((index >= 0) && (index < this.getPanelCount())) {
			return (ArffPanel) this.tabbedPane.getComponentAt(index);
		} else {
			return null;
		}
	}

	/**
	 * returns the currently selected tab index
	 *
	 * @return the index of the currently selected tab
	 */
	public int getCurrentIndex() {
		return this.tabbedPane.getSelectedIndex();
	}

	/**
	 * returns the currently selected panel
	 *
	 * @return the currently selected panel
	 */
	public ArffPanel getCurrentPanel() {
		return this.getPanel(this.getCurrentIndex());
	}

	/**
	 * checks whether a panel is currently selected
	 *
	 * @return true if a panel is currently selected
	 */
	public boolean isPanelSelected() {
		return (this.getCurrentPanel() != null);
	}

	/**
	 * returns the filename of the specified panel
	 *
	 * @param index the index of the panel
	 * @return the filename for the panel
	 */
	public String getFilename(final int index) {
		String result;
		ArffPanel panel;

		result = "";
		panel = this.getPanel(index);

		if (panel != null) {
			result = panel.getFilename();
		}

		return result;
	}

	/**
	 * returns the filename of the current tab
	 *
	 * @return the current filename
	 */
	public String getCurrentFilename() {
		return this.getFilename(this.getCurrentIndex());
	}

	/**
	 * sets the filename of the specified panel
	 *
	 * @param index the index of the panel
	 * @param filename the new filename
	 */
	public void setFilename(final int index, final String filename) {
		ArffPanel panel;

		panel = this.getPanel(index);

		if (panel != null) {
			panel.setFilename(filename);
			this.setTabTitle(panel);
		}
	}

	/**
	 * sets the filename of the current tab
	 *
	 * @param filename the new filename
	 */
	public void setCurrentFilename(final String filename) {
		this.setFilename(this.getCurrentIndex(), filename);
	}

	/**
	 * if the file is changed it pops up a dialog whether to change the settings.
	 * if the project wasn't changed or saved it returns TRUE
	 *
	 * @return true if project wasn't changed or saved
	 */
	protected boolean saveChanges() {
		return this.saveChanges(true);
	}

	/**
	 * if the file is changed it pops up a dialog whether to change the settings.
	 * if the project wasn't changed or saved it returns TRUE
	 *
	 * @param showCancel whether we have YES/NO/CANCEL or only YES/NO
	 * @return true if project wasn't changed or saved
	 */
	protected boolean saveChanges(final boolean showCancel) {
		int button;
		boolean result;

		if (!this.isPanelSelected()) {
			return true;
		}

		result = !this.getCurrentPanel().isChanged();

		if (this.getCurrentPanel().isChanged()) {
			try {
				if (showCancel) {
					button = ComponentHelper.showMessageBox(this, "Changed", "The file is not saved - Do you want to save it?", JOptionPane.YES_NO_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
				} else {
					button = ComponentHelper.showMessageBox(this, "Changed", "The file is not saved - Do you want to save it?", JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE);
				}
			} catch (Exception e) {
				button = JOptionPane.CANCEL_OPTION;
			}

			switch (button) {
			case JOptionPane.YES_OPTION:
				this.saveFile();
				result = !this.getCurrentPanel().isChanged();
				break;
			case JOptionPane.NO_OPTION:
				result = true;
				break;
			case JOptionPane.CANCEL_OPTION:
				result = false;
				break;
			}
		}

		return result;
	}

	/**
	 * loads the specified file
	 *
	 * @param filename the file to load
	 * @param loaders optional varargs loader to use
	 */
	public void loadFile(final String filename, final AbstractFileLoader... loaders) {
		ArffPanel panel;

		panel = new ArffPanel(filename, loaders);
		panel.addChangeListener(this);
		this.tabbedPane.addTab(panel.getTitle(), panel);
		this.tabbedPane.setSelectedIndex(this.tabbedPane.getTabCount() - 1);
	}

	/**
	 * loads the specified file into the table
	 */
	public void loadFile() {
		int retVal;
		int i;
		String filename;

		retVal = this.fileChooser.showOpenDialog(this);
		if (retVal != ConverterFileChooser.APPROVE_OPTION) {
			return;
		}

		this.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));

		for (i = 0; i < this.fileChooser.getSelectedFiles().length; i++) {
			filename = this.fileChooser.getSelectedFiles()[i].getAbsolutePath();
			this.loadFile(filename, this.fileChooser.getLoader());
		}

		this.setCursor(Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
	}

	/**
	 * saves the current data into a file
	 */
	public void saveFile() {
		ArffPanel panel;
		String filename;
		AbstractSaver saver;

		// no panel? -> exit
		panel = this.getCurrentPanel();
		if (panel == null) {
			return;
		}

		filename = panel.getFilename();

		if (filename.equals(ArffPanel.TAB_INSTANCES)) {
			this.saveFileAs();
		} else {
			saver = ConverterUtils.getSaverForFile(filename);
			try {
				saver.setFile(new File(filename));
				saver.setInstances(panel.getInstances());
				saver.writeBatch();
				panel.setChanged(false);
				this.setCurrentFilename(filename);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * saves the current data into a new file
	* @throws InterruptedException
	 */
	public void saveFileAs() {
		int retVal;
		ArffPanel panel;

		// no panel? -> exit
		panel = this.getCurrentPanel();
		if (panel == null) {
			System.out.println("nothing selected!");
			return;
		}

		if (!this.getCurrentFilename().equals("")) {
			try {
				this.fileChooser.setSelectedFile(new File(this.getCurrentFilename()));
			} catch (Exception e) {
				// ignore
			}
		}

		// set filter for savers
		try {
			this.fileChooser.setCapabilitiesFilter(Capabilities.forInstances(panel.getInstances()));
		} catch (Exception e) {
			this.fileChooser.setCapabilitiesFilter(null);
		}

		retVal = this.fileChooser.showSaveDialog(this);
		if (retVal != ConverterFileChooser.APPROVE_OPTION) {
			return;
		}

		panel.setChanged(false);
		this.setCurrentFilename(this.fileChooser.getSelectedFile().getAbsolutePath());
		// saveFile();

		AbstractFileSaver saver = this.fileChooser.getSaver();
		try {
			saver.setInstances(panel.getInstances());
			saver.writeBatch();
			panel.setChanged(false);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	/**
	 * closes the current tab
	 */
	public void closeFile() {
		this.closeFile(true);
	}

	/**
	 * closes the current tab
	 *
	 * @param showCancel whether to show an additional CANCEL button in the
	 *          "Want to save changes"-dialog
	 * @see #saveChanges(boolean)
	 */
	public void closeFile(final boolean showCancel) {
		if (this.getCurrentIndex() == -1) {
			return;
		}

		if (!this.saveChanges(showCancel)) {
			return;
		}

		this.tabbedPane.removeTabAt(this.getCurrentIndex());
		this.updateFrameTitle();
		System.gc();
	}

	/**
	 * closes all open files
	 */
	public void closeAllFiles() {
		while (this.tabbedPane.getTabCount() > 0) {
			if (!this.saveChanges(true)) {
				return;
			}

			this.tabbedPane.removeTabAt(this.getCurrentIndex());
			this.updateFrameTitle();
			System.gc();
		}
	}

	/**
	 * displays some properties of the instances
	 */
	public void showProperties() {
		ArffPanel panel;
		ListSelectorDialog dialog;
		Vector<String> props;
		Instances inst;

		panel = this.getCurrentPanel();
		if (panel == null) {
			return;
		}

		inst = panel.getInstances();
		if (inst == null) {
			return;
		}
		if (inst.classIndex() < 0) {
			inst.setClassIndex(inst.numAttributes() - 1);
		}

		// get some data
		props = new Vector<String>();
		props.add("Filename: " + panel.getFilename());
		props.add("Relation name: " + inst.relationName());
		props.add("# of instances: " + inst.numInstances());
		props.add("# of attributes: " + inst.numAttributes());
		props.add("Class attribute: " + inst.classAttribute().name());
		props.add("# of class labels: " + inst.numClasses());

		dialog = new ListSelectorDialog(this.getParentFrame(), new JList(props));
		dialog.showDialog();
	}

	/**
	 * closes the window, i.e., if the parent is not null and implements the
	 * WindowListener interface it calls the windowClosing method
	 */
	public void close() {
		if (this.getParentInternalFrame() != null) {
			this.getParentInternalFrame().doDefaultCloseAction();
		} else if (this.getParentFrame() != null) {
			((Window) this.getParentFrame()).dispatchEvent(new WindowEvent(this.getParentFrame(), WindowEvent.WINDOW_CLOSING));
		}
	}

	/**
	 * undoes the last action
	 */
	public void undo() {
		if (!this.isPanelSelected()) {
			return;
		}

		this.getCurrentPanel().undo();
	}

	/**
	 * copies the content of the selection to the clipboard
	 */
	public void copyContent() {
		if (!this.isPanelSelected()) {
			return;
		}

		this.getCurrentPanel().copyContent();
	}

	/**
	 * searches for a string in the cells
	 */
	public void search() {
		if (!this.isPanelSelected()) {
			return;
		}

		this.getCurrentPanel().search();
	}

	/**
	 * clears the search, i.e. resets the found cells
	 */
	public void clearSearch() {
		if (!this.isPanelSelected()) {
			return;
		}

		this.getCurrentPanel().clearSearch();
	}

	/**
	 * renames the current selected Attribute
	 */
	public void renameAttribute() {
		if (!this.isPanelSelected()) {
			return;
		}

		this.getCurrentPanel().renameAttribute();
	}

	/**
	 * sets the current selected Attribute as class attribute, i.e. it moves it to
	 * the end of the attributes
	 */
	public void attributeAsClass() {
		if (!this.isPanelSelected()) {
			return;
		}

		this.getCurrentPanel().attributeAsClass();
	}

	/**
	 * deletes the current selected Attribute or several chosen ones
	 *
	 * @param multiple whether to delete myultiple attributes
	 */
	public void deleteAttribute(final boolean multiple) {
		if (!this.isPanelSelected()) {
			return;
		}

		if (multiple) {
			this.getCurrentPanel().deleteAttributes();
		} else {
			this.getCurrentPanel().deleteAttribute();
		}
	}

	/**
	 * deletes the current selected Instance or several chosen ones
	 *
	 * @param multiple whether to delete multiple instances
	 */
	public void deleteInstance(final boolean multiple) {
		if (!this.isPanelSelected()) {
			return;
		}

		if (multiple) {
			this.getCurrentPanel().deleteInstances();
		} else {
			this.getCurrentPanel().deleteInstance();
		}
	}

	/**
	 * sorts the current selected attribute
	 */
	public void sortInstances() {
		if (!this.isPanelSelected()) {
			return;
		}

		this.getCurrentPanel().sortInstances();
	}

	/**
	 * displays all the attributes, returns the selected item or NULL if canceled
	 *
	 * @return the name of the selected attribute
	 */
	public String showAttributes() {
		ArffSortedTableModel model;
		ListSelectorDialog dialog;
		int i;
		JList list;
		String name;
		int result;

		if (!this.isPanelSelected()) {
			return null;
		}

		list = new JList(this.getCurrentPanel().getAttributes());
		dialog = new ListSelectorDialog(this.getParentFrame(), list);
		result = dialog.showDialog();

		if (result == ListSelectorDialog.APPROVE_OPTION) {
			model = (ArffSortedTableModel) this.getCurrentPanel().getTable().getModel();
			name = list.getSelectedValue().toString();
			i = model.getAttributeColumn(name);
			JTableHelper.scrollToVisible(this.getCurrentPanel().getTable(), 0, i);
			this.getCurrentPanel().getTable().setSelectedColumn(i);
			return name;
		} else {
			return null;
		}
	}

	/**
	 * displays all the distinct values for an attribute
	 */
	public void showValues() {
		String attribute;
		ArffSortedTableModel model;
		ArffTable table;
		HashSet<String> values;
		Vector<String> items;
		Iterator<String> iter;
		ListSelectorDialog dialog;
		int i;
		int col;

		// choose attribute to retrieve values for
		attribute = this.showAttributes();
		if (attribute == null) {
			return;
		}

		table = this.getCurrentPanel().getTable();
		model = (ArffSortedTableModel) table.getModel();

		// get column index
		col = -1;
		for (i = 0; i < table.getColumnCount(); i++) {
			if (table.getPlainColumnName(i).equals(attribute)) {
				col = i;
				break;
			}
		}
		// not found?
		if (col == -1) {
			return;
		}

		// get values
		values = new HashSet<String>();
		items = new Vector<String>();
		for (i = 0; i < model.getRowCount(); i++) {
			values.add(model.getValueAt(i, col).toString());
		}
		if (values.isEmpty()) {
			return;
		}
		iter = values.iterator();
		while (iter.hasNext()) {
			items.add(iter.next());
		}
		Collections.sort(items);

		dialog = new ListSelectorDialog(this.getParentFrame(), new JList(items));
		dialog.showDialog();
	}

	/**
	 * sets the optimal column width for all columns
	 */
	public void setOptimalColWidths() {
		if (!this.isPanelSelected()) {
			return;
		}

		this.getCurrentPanel().setOptimalColWidths();
	}

	/**
	 * invoked when an action occurs
	 *
	 * @param e the action event
	 */
	@Override
	public void actionPerformed(final ActionEvent e) {
		Object o;

		o = e.getSource();

		if (o == this.menuFileOpen) {
			this.loadFile();
		} else if (o == this.menuFileSave) {
			this.saveFile();
		} else if (o == this.menuFileSaveAs) {
			this.saveFileAs();
		} else if (o == this.menuFileClose) {
			this.closeFile();
		} else if (o == this.menuFileCloseAll) {
			this.closeAllFiles();
		} else if (o == this.menuFileProperties) {
			this.showProperties();
		} else if (o == this.menuFileExit) {
			this.close();
		} else if (o == this.menuEditUndo) {
			this.undo();
		} else if (o == this.menuEditCopy) {
			this.copyContent();
		} else if (o == this.menuEditSearch) {
			this.search();
		} else if (o == this.menuEditClearSearch) {
			this.clearSearch();
		} else if (o == this.menuEditDeleteAttribute) {
			this.deleteAttribute(false);
		} else if (o == this.menuEditDeleteAttributes) {
			this.deleteAttribute(true);
		} else if (o == this.menuEditRenameAttribute) {
			this.renameAttribute();
		} else if (o == this.menuEditAttributeAsClass) {
			this.attributeAsClass();
		} else if (o == this.menuEditDeleteInstance) {
			this.deleteInstance(false);
		} else if (o == this.menuEditDeleteInstances) {
			this.deleteInstance(true);
		} else if (o == this.menuEditSortInstances) {
			this.sortInstances();
		} else if (o == this.menuViewAttributes) {
			this.showAttributes();
		} else if (o == this.menuViewValues) {
			this.showValues();
		} else if (o == this.menuViewOptimalColWidths) {
			this.setOptimalColWidths();
		}

		this.updateMenu();
	}

	/**
	 * Invoked when the target of the listener has changed its state.
	 *
	 * @param e the change event
	 */
	@Override
	public void stateChanged(final ChangeEvent e) {
		this.updateFrameTitle();
		this.updateMenu();

		// did the content of panel change? -> change title of tab
		if (e.getSource() instanceof JComponent) {
			this.setTabTitle((JComponent) e.getSource());
		}
	}

	/**
	 * returns only the classname
	 *
	 * @return the classname
	 */
	@Override
	public String toString() {
		return this.getClass().getName();
	}
}
