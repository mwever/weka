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
 *    MatrixPanel.java
 *    Copyright (C) 2002-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.gui.visualize;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dialog.ModalityType;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Image;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSlider;
import javax.swing.JSplitPane;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import weka.core.Attribute;
import weka.core.Environment;
import weka.core.Instances;
import weka.core.Settings;
import weka.core.Utils;
import weka.gui.ExtensionFileFilter;

/**
 * This panel displays a plot matrix of the user selected attributes of a given
 * data set.
 *
 * The datapoints are coloured using a discrete colouring set if the user has
 * selected a nominal attribute for colouring. If the user has selected a
 * numeric attribute then the datapoints are coloured using a colour spectrum
 * ranging from blue to red (low values to high). Datapoints missing a class
 * value are displayed in black.
 *
 * @author Ashraf M. Kibriya (amk14@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class MatrixPanel extends JPanel {

	/** for serialization */
	private static final long serialVersionUID = -1232642719869188740L;

	/** The that panel contains the actual matrix */
	private final Plot m_plotsPanel;

	/** The panel that displays the legend of the colouring attribute */
	protected final ClassPanel m_cp = new ClassPanel();

	/**
	 * The panel that contains all the buttons and tools, i.e. resize, jitter bars
	 * and sub-sampling buttons etc on the bottom of the panel
	 */
	protected JPanel optionsPanel;

	/** Split pane for splitting the matrix and the buttons and bars */
	protected JSplitPane jp;
	/**
	 * The button that updates the display to reflect the changes made by the
	 * user. E.g. changed attribute set for the matrix
	 */
	protected JButton m_updateBt = new JButton("Update");

	/** The button to display a window to select attributes */
	protected JButton m_selAttrib = new JButton("Select Attributes");

	/** The dataset for which this panel will display the plot matrix for */
	protected Instances m_data = null;

	/** The list for selecting the attributes to display the plot matrix */
	protected JList m_attribList = new JList();

	/** The scroll pane to scrolling the matrix */
	protected final JScrollPane m_js = new JScrollPane();

	/** The combo box to allow user to select the colouring attribute */
	protected JComboBox m_classAttrib = new JComboBox();

	/** The slider to adjust the size of the cells in the matrix */
	protected JSlider m_plotSize = new JSlider(50, 200, 100);

	/** The slider to adjust the size of the datapoints */
	protected JSlider m_pointSize = new JSlider(1, 10, 1);

	/** The slider to add jitter to the plots */
	protected JSlider m_jitter = new JSlider(0, 20, 0);

	/** For adding random jitter */
	private final Random rnd = new Random();

	/** Array containing precalculated jitter values */
	private int jitterVals[][];

	/** This stores the size of the datapoint */
	private int datapointSize = 1;

	/** The text area for percentage to resample data */
	protected JTextField m_resamplePercent = new JTextField(5);

	/** The label for resample percentage */
	protected JButton m_resampleBt = new JButton("SubSample % :");

	/** Random seed for random subsample */
	protected JTextField m_rseed = new JTextField(5);

	/** Displays the current size beside the slider bar for cell size */
	private final JLabel m_plotSizeLb = new JLabel("PlotSize: [100]");

	/** Displays the current size beside the slider bar for point size */
	private final JLabel m_pointSizeLb = new JLabel("PointSize: [10]");

	/** This array contains the indices of the attributes currently selected */
	private int[] m_selectedAttribs;

	/** This contains the index of the currently selected colouring attribute */
	private int m_classIndex;

	/**
	 * This is a local array cache for all the instance values for faster
	 * rendering
	 */
	private int[][] m_points;

	/**
	 * This is an array cache for the colour of each of the instances depending on
	 * the colouring attribute. If the colouring attribute is nominal then it
	 * contains the index of the colour in our colour list. Otherwise, for numeric
	 * colouring attribute, it contains the precalculated red component for each
	 * instance's colour
	 */
	private int[] m_pointColors;

	/**
	 * Contains true for each attribute value (only the selected attributes+class
	 * attribute) that is missing, for each instance. m_missing[i][j] == true if
	 * m_selectedAttribs[j] is missing in instance i.
	 * m_missing[i][m_missing[].length-1] == true if class value is missing in
	 * instance i.
	 */
	private boolean[][] m_missing;

	/**
	 * This array contains for the classAttribute: <br>
	 * m_type[0] = [type of attribute, nominal, string or numeric]<br>
	 * m_type[1] = [number of discrete values of nominal or string attribute <br>
	 * or same as m_type[0] for numeric attribute]
	 */
	private int[] m_type;

	/** Stores the maximum size for PlotSize label to keep it's size constant */
	private final Dimension m_plotLBSizeD;

	/** Stores the maximum size for PointSize label to keep it's size constant */
	private final Dimension m_pointLBSizeD;

	/** Contains discrete colours for colouring for nominal attributes */
	private final ArrayList<Color> m_colorList = new ArrayList<Color>();

	/** default colour list */
	private static final Color[] m_defaultColors = { Color.blue, Color.red, Color.cyan, new Color(75, 123, 130), Color.pink, Color.green, Color.orange, new Color(255, 0, 255), new Color(255, 0, 0), new Color(0, 255, 0), Color.black };

	/** color for the font used in column and row names */
	private final Color fontColor = new Color(98, 101, 156);

	/** font used in column and row names */
	private final java.awt.Font f = new java.awt.Font("Dialog", java.awt.Font.BOLD, 11);

	/** Settings (if available) to pass through to the VisualizePanels */
	protected Settings m_settings;

	/** For the background of the little plots */
	protected Color m_backgroundColor = Color.white;

	/**
	 * ID of the owner (perspective, panel etc.) under which to lookup our
	 * settings
	 */
	protected String m_settingsOwnerID;

	protected transient Image m_osi = null;
	protected boolean[][] m_plottedCells;
	protected boolean m_regenerateOSI = true;
	protected boolean m_clearOSIPlottedCells;
	protected double m_previousPercent = -1;

	protected JCheckBox m_fastScroll = new JCheckBox("Fast scrolling (uses more memory)");

	/**
	 * Constructor
	 */
	public MatrixPanel() {
		this.m_rseed.setText("1");

		/** Setting up GUI **/
		this.m_selAttrib.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(final ActionEvent ae) {
				final JDialog jd = new JDialog((JFrame) MatrixPanel.this.getTopLevelAncestor(), "Attribute Selection Panel", ModalityType.DOCUMENT_MODAL);

				JPanel jp = new JPanel();
				JScrollPane js = new JScrollPane(MatrixPanel.this.m_attribList);
				JButton okBt = new JButton("OK");
				JButton cancelBt = new JButton("Cancel");
				final int[] savedSelection = MatrixPanel.this.m_attribList.getSelectedIndices();

				okBt.addActionListener(new ActionListener() {
					@Override
					public void actionPerformed(final ActionEvent e) {
						jd.dispose();
					}
				});

				cancelBt.addActionListener(new ActionListener() {
					@Override
					public void actionPerformed(final ActionEvent e) {
						MatrixPanel.this.m_attribList.setSelectedIndices(savedSelection);
						jd.dispose();
					}
				});
				jd.addWindowListener(new WindowAdapter() {
					@Override
					public void windowClosing(final WindowEvent e) {
						MatrixPanel.this.m_attribList.setSelectedIndices(savedSelection);
						jd.dispose();
					}
				});
				jp.add(okBt);
				jp.add(cancelBt);

				jd.getContentPane().add(js, BorderLayout.CENTER);
				jd.getContentPane().add(jp, BorderLayout.SOUTH);

				if (js.getPreferredSize().width < 200) {
					jd.setSize(250, 250);
				} else {
					jd.setSize(js.getPreferredSize().width + 10, 250);
				}

				jd.setLocation(MatrixPanel.this.m_selAttrib.getLocationOnScreen().x, MatrixPanel.this.m_selAttrib.getLocationOnScreen().y - jd.getHeight());
				jd.setVisible(true);
			}
		});

		this.m_updateBt.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(final ActionEvent e) {
				MatrixPanel.this.updatePanel();
			}
		});
		this.m_updateBt.setPreferredSize(this.m_selAttrib.getPreferredSize());

		this.m_jitter.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(final ChangeEvent ce) {
				if (MatrixPanel.this.m_fastScroll.isSelected()) {
					MatrixPanel.this.m_clearOSIPlottedCells = true;
				}
			}
		});

		this.m_plotSize.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(final ChangeEvent ce) {
				MatrixPanel.this.m_plotSizeLb.setText("PlotSize: [" + MatrixPanel.this.m_plotSize.getValue() + "]");
				MatrixPanel.this.m_plotSizeLb.setPreferredSize(MatrixPanel.this.m_plotLBSizeD);
				MatrixPanel.this.m_jitter.setMaximum(MatrixPanel.this.m_plotSize.getValue() / 5); // 20% of cell Size
				MatrixPanel.this.m_regenerateOSI = true;
			}
		});

		this.m_pointSize.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(final ChangeEvent ce) {
				MatrixPanel.this.m_pointSizeLb.setText("PointSize: [" + MatrixPanel.this.m_pointSize.getValue() + "]");
				MatrixPanel.this.m_pointSizeLb.setPreferredSize(MatrixPanel.this.m_pointLBSizeD);
				MatrixPanel.this.datapointSize = MatrixPanel.this.m_pointSize.getValue();
				if (MatrixPanel.this.m_fastScroll.isSelected()) {
					MatrixPanel.this.m_clearOSIPlottedCells = true;
				}
			}
		});

		this.m_resampleBt.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(final ActionEvent e) {
				JLabel rseedLb = new JLabel("Random Seed: ");
				JTextField rseedTxt = MatrixPanel.this.m_rseed;
				JLabel percentLb = new JLabel("Subsample as");
				JLabel percent2Lb = new JLabel("% of input: ");
				final JTextField percentTxt = new JTextField(5);
				percentTxt.setText(MatrixPanel.this.m_resamplePercent.getText());
				JButton doneBt = new JButton("Done");

				final JDialog jd = new JDialog((JFrame) MatrixPanel.this.getTopLevelAncestor(), "Subsample % Panel", ModalityType.DOCUMENT_MODAL) {
					private static final long serialVersionUID = -269823533147146296L;

					@Override
					public void dispose() {
						MatrixPanel.this.m_resamplePercent.setText(percentTxt.getText());
						super.dispose();
					}
				};
				jd.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);

				doneBt.addActionListener(new ActionListener() {
					@Override
					public void actionPerformed(final ActionEvent ae) {
						jd.dispose();
					}
				});
				GridBagLayout gbl = new GridBagLayout();
				GridBagConstraints gbc = new GridBagConstraints();
				JPanel p1 = new JPanel(gbl);
				gbc.anchor = GridBagConstraints.WEST;
				gbc.fill = GridBagConstraints.HORIZONTAL;
				gbc.insets = new Insets(0, 2, 2, 2);
				gbc.gridwidth = GridBagConstraints.RELATIVE;
				p1.add(rseedLb, gbc);
				gbc.weightx = 0;
				gbc.gridwidth = GridBagConstraints.REMAINDER;
				gbc.weightx = 1;
				p1.add(rseedTxt, gbc);
				gbc.insets = new Insets(8, 2, 0, 2);
				gbc.weightx = 0;
				p1.add(percentLb, gbc);
				gbc.insets = new Insets(0, 2, 2, 2);
				gbc.gridwidth = GridBagConstraints.RELATIVE;
				p1.add(percent2Lb, gbc);
				gbc.gridwidth = GridBagConstraints.REMAINDER;
				gbc.weightx = 1;
				p1.add(percentTxt, gbc);
				gbc.insets = new Insets(8, 2, 2, 2);

				JPanel p3 = new JPanel(gbl);
				gbc.fill = GridBagConstraints.HORIZONTAL;
				gbc.gridwidth = GridBagConstraints.REMAINDER;
				gbc.weightx = 1;
				gbc.weighty = 0;
				p3.add(p1, gbc);
				gbc.insets = new Insets(8, 4, 8, 4);
				p3.add(doneBt, gbc);

				jd.getContentPane().setLayout(new BorderLayout());
				jd.getContentPane().add(p3, BorderLayout.NORTH);
				jd.pack();
				jd.setLocation(MatrixPanel.this.m_resampleBt.getLocationOnScreen().x, MatrixPanel.this.m_resampleBt.getLocationOnScreen().y - jd.getHeight());
				jd.setVisible(true);
			}
		});

		this.optionsPanel = new JPanel(new GridBagLayout()); // all the rest of the
		// panels are in here.
		final JPanel p2 = new JPanel(new BorderLayout()); // this has class colour
															// panel
		final JPanel p3 = new JPanel(new GridBagLayout()); // this has update and
															// select buttons
		final JPanel p4 = new JPanel(new GridBagLayout()); // this has the slider
															// bars and combobox
		GridBagConstraints gbc = new GridBagConstraints();

		this.m_plotLBSizeD = this.m_plotSizeLb.getPreferredSize();
		this.m_pointLBSizeD = this.m_pointSizeLb.getPreferredSize();
		this.m_pointSizeLb.setText("PointSize: [1]");
		this.m_pointSizeLb.setPreferredSize(this.m_pointLBSizeD);
		this.m_resampleBt.setPreferredSize(this.m_selAttrib.getPreferredSize());

		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.anchor = GridBagConstraints.NORTHWEST;
		gbc.insets = new Insets(2, 2, 2, 2);
		p4.add(this.m_plotSizeLb, gbc);
		gbc.weightx = 1;
		gbc.gridwidth = GridBagConstraints.REMAINDER;
		p4.add(this.m_plotSize, gbc);
		gbc.weightx = 0;
		gbc.gridwidth = GridBagConstraints.RELATIVE;
		p4.add(this.m_pointSizeLb, gbc);
		gbc.weightx = 1;
		gbc.gridwidth = GridBagConstraints.REMAINDER;
		p4.add(this.m_pointSize, gbc);
		gbc.weightx = 0;
		gbc.gridwidth = GridBagConstraints.RELATIVE;
		p4.add(new JLabel("Jitter: "), gbc);
		gbc.weightx = 1;
		gbc.gridwidth = GridBagConstraints.REMAINDER;
		p4.add(this.m_jitter, gbc);
		p4.add(this.m_classAttrib, gbc);

		gbc.gridwidth = GridBagConstraints.REMAINDER;
		gbc.weightx = 1;
		gbc.fill = GridBagConstraints.NONE;
		p3.add(this.m_fastScroll, gbc);
		p3.add(this.m_updateBt, gbc);
		p3.add(this.m_selAttrib, gbc);
		gbc.gridwidth = GridBagConstraints.RELATIVE;
		gbc.weightx = 0;
		gbc.fill = GridBagConstraints.VERTICAL;
		gbc.anchor = GridBagConstraints.WEST;
		p3.add(this.m_resampleBt, gbc);
		gbc.gridwidth = GridBagConstraints.REMAINDER;
		p3.add(this.m_resamplePercent, gbc);

		p2.setBorder(BorderFactory.createTitledBorder("Class Colour"));
		p2.add(this.m_cp, BorderLayout.SOUTH);

		gbc.insets = new Insets(8, 5, 2, 5);
		gbc.anchor = GridBagConstraints.SOUTHWEST;
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.weightx = 1;
		gbc.gridwidth = GridBagConstraints.RELATIVE;
		this.optionsPanel.add(p4, gbc);
		gbc.gridwidth = GridBagConstraints.REMAINDER;
		this.optionsPanel.add(p3, gbc);
		this.optionsPanel.add(p2, gbc);

		this.m_fastScroll.setSelected(false);
		this.m_fastScroll.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(final ActionEvent e) {
				if (!MatrixPanel.this.m_fastScroll.isSelected()) {
					MatrixPanel.this.m_osi = null;
				} else {
					MatrixPanel.this.m_plottedCells = new boolean[MatrixPanel.this.m_selectedAttribs.length][MatrixPanel.this.m_selectedAttribs.length];
				}
				MatrixPanel.this.invalidate();
				MatrixPanel.this.repaint();
			}
		});

		this.addComponentListener(new ComponentAdapter() {
			@Override
			public void componentResized(final ComponentEvent cv) {
				MatrixPanel.this.m_js.setMinimumSize(new Dimension(MatrixPanel.this.getWidth(), MatrixPanel.this.getHeight() - MatrixPanel.this.optionsPanel.getPreferredSize().height - 10));
				MatrixPanel.this.jp.setDividerLocation(MatrixPanel.this.getHeight() - MatrixPanel.this.optionsPanel.getPreferredSize().height - 10);
			}
		});

		this.optionsPanel.setMinimumSize(new Dimension(0, 0));
		this.jp = new JSplitPane(JSplitPane.VERTICAL_SPLIT, this.m_js, this.optionsPanel);
		this.jp.setOneTouchExpandable(true);
		this.jp.setResizeWeight(1);
		this.setLayout(new BorderLayout());
		this.add(this.jp, BorderLayout.CENTER);

		/** Setting up the initial color list **/
		for (int i = 0; i < m_defaultColors.length; i++) {
			this.m_colorList.add(m_defaultColors[i]);
		}

		/** Initializing internal fields and components **/
		this.m_selectedAttribs = this.m_attribList.getSelectedIndices();
		this.m_plotsPanel = new Plot();
		this.m_plotsPanel.setLayout(null);
		this.m_js.getHorizontalScrollBar().setUnitIncrement(10);
		this.m_js.getVerticalScrollBar().setUnitIncrement(10);
		this.m_js.setViewportView(this.m_plotsPanel);
		this.m_js.setColumnHeaderView(this.m_plotsPanel.getColHeader());
		this.m_js.setRowHeaderView(this.m_plotsPanel.getRowHeader());
		final JLabel lb = new JLabel(" Plot Matrix");
		lb.setFont(this.f);
		lb.setForeground(this.fontColor);
		lb.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
		this.m_js.setCorner(JScrollPane.UPPER_LEFT_CORNER, lb);
		this.m_cp.setInstances(this.m_data);
		this.m_cp.setBorder(BorderFactory.createEmptyBorder(15, 10, 10, 10));
		this.m_cp.addRepaintNotify(this.m_plotsPanel);
		// m_updateBt.doClick(); //not until setting up the instances
	}

	/**
	 * Initializes internal data fields, i.e. data values, type, missing and color
	 * cache arrays
	 */
	public void initInternalFields() {
		Instances inst = this.m_data;
		this.m_classIndex = this.m_classAttrib.getSelectedIndex();
		this.m_selectedAttribs = this.m_attribList.getSelectedIndices();
		double minC = 0, maxC = 0;

		/** Resampling **/
		double currentPercent = Double.parseDouble(this.m_resamplePercent.getText());
		if (currentPercent <= 100) {
			if (currentPercent != this.m_previousPercent) {
				this.m_clearOSIPlottedCells = true;
			}
			inst = new Instances(this.m_data, 0, this.m_data.numInstances());
			try {
				inst.randomize(new Random(Integer.parseInt(this.m_rseed.getText())));
			} catch (InterruptedException e) {
				throw new IllegalStateException(e);
			}

			// System.err.println("gettingPercent: " +
			// Math.round(
			// Double.parseDouble(m_resamplePercent.getText())
			// / 100D * m_data.numInstances()
			// )
			// );

			inst = new Instances(inst, 0, (int) Math.round(currentPercent / 100D * inst.numInstances()));
			this.m_previousPercent = currentPercent;
		}
		this.m_points = new int[inst.numInstances()][this.m_selectedAttribs.length]; // changed
		this.m_pointColors = new int[inst.numInstances()];
		this.m_missing = new boolean[inst.numInstances()][this.m_selectedAttribs.length + 1]; // changed
		this.m_type = new int[2]; // [m_selectedAttribs.length]; //changed
		this.jitterVals = new int[inst.numInstances()][2];

		/**
		 * Setting up the color list for non-numeric attribute as well as jittervals
		 **/
		if (!(inst.attribute(this.m_classIndex).isNumeric())) {

			for (int i = this.m_colorList.size(); i < inst.attribute(this.m_classIndex).numValues() + 1; i++) {
				Color pc = m_defaultColors[i % 10];
				int ija = i / 10;
				ija *= 2;
				for (int j = 0; j < ija; j++) {
					pc = pc.darker();
				}
				this.m_colorList.add(pc);
			}

			for (int i = 0; i < inst.numInstances(); i++) {
				// set to black for missing class value which is last colour is default
				// list
				if (inst.instance(i).isMissing(this.m_classIndex)) {
					this.m_pointColors[i] = m_defaultColors.length - 1;
				} else {
					this.m_pointColors[i] = (int) inst.instance(i).value(this.m_classIndex);
				}

				this.jitterVals[i][0] = this.rnd.nextInt(this.m_jitter.getValue() + 1) - this.m_jitter.getValue() / 2;
				this.jitterVals[i][1] = this.rnd.nextInt(this.m_jitter.getValue() + 1) - this.m_jitter.getValue() / 2;

			}
		}
		/** Setting up color variations for numeric attribute as well as jittervals **/
		else {
			for (int i = 0; i < inst.numInstances(); i++) {
				if (!(inst.instance(i).isMissing(this.m_classIndex))) {
					minC = maxC = inst.instance(i).value(this.m_classIndex);
					break;
				}
			}

			for (int i = 1; i < inst.numInstances(); i++) {
				if (!(inst.instance(i).isMissing(this.m_classIndex))) {
					if (minC > inst.instance(i).value(this.m_classIndex)) {
						minC = inst.instance(i).value(this.m_classIndex);
					}
					if (maxC < inst.instance(i).value(this.m_classIndex)) {
						maxC = inst.instance(i).value(this.m_classIndex);
					}
				}
			}

			for (int i = 0; i < inst.numInstances(); i++) {
				double r = (inst.instance(i).value(this.m_classIndex) - minC) / (maxC - minC);
				r = (r * 240) + 15;
				this.m_pointColors[i] = (int) r;

				this.jitterVals[i][0] = this.rnd.nextInt(this.m_jitter.getValue() + 1) - this.m_jitter.getValue() / 2;
				this.jitterVals[i][1] = this.rnd.nextInt(this.m_jitter.getValue() + 1) - this.m_jitter.getValue() / 2;
			}
		}

		/** Creating local cache of the data values **/
		double min[] = new double[this.m_selectedAttribs.length], max = 0; // changed
		double ratio[] = new double[this.m_selectedAttribs.length]; // changed
		double cellSize = this.m_plotSize.getValue(), temp1 = 0, temp2 = 0;

		for (int j = 0; j < this.m_selectedAttribs.length; j++) {
			int i;
			for (i = 0; i < inst.numInstances(); i++) {
				min[j] = max = 0;
				if (!(inst.instance(i).isMissing(this.m_selectedAttribs[j]))) {
					min[j] = max = inst.instance(i).value(this.m_selectedAttribs[j]);
					break;
				}
			}
			for (; i < inst.numInstances(); i++) {
				if (!(inst.instance(i).isMissing(this.m_selectedAttribs[j]))) {
					if (inst.instance(i).value(this.m_selectedAttribs[j]) < min[j]) {
						min[j] = inst.instance(i).value(this.m_selectedAttribs[j]);
					}
					if (inst.instance(i).value(this.m_selectedAttribs[j]) > max) {
						max = inst.instance(i).value(this.m_selectedAttribs[j]);
					}
				}
			}
			ratio[j] = cellSize / (max - min[j]);
		}

		boolean classIndexProcessed = false;
		for (int j = 0; j < this.m_selectedAttribs.length; j++) {
			if (inst.attribute(this.m_selectedAttribs[j]).isNominal() || inst.attribute(this.m_selectedAttribs[j]).isString()) {
				// m_type[0][j] = 1; m_type[1][j] =
				// inst.attribute(m_selectedAttribs[j]).numValues();

				temp1 = cellSize / inst.attribute(this.m_selectedAttribs[j]).numValues(); // m_type[1][j];
				temp2 = temp1 / 2;
				for (int i = 0; i < inst.numInstances(); i++) {
					this.m_points[i][j] = (int) Math.round(temp2 + temp1 * inst.instance(i).value(this.m_selectedAttribs[j]));
					if (inst.instance(i).isMissing(this.m_selectedAttribs[j])) {
						this.m_missing[i][j] = true; // represents missing value
						if (this.m_selectedAttribs[j] == this.m_classIndex) {
							this.m_missing[i][this.m_missing[0].length - 1] = true;
							classIndexProcessed = true;
						}
					}
				}
			} else {
				// m_type[0][j] = m_type[1][j] = 0;
				for (int i = 0; i < inst.numInstances(); i++) {
					this.m_points[i][j] = (int) Math.round((inst.instance(i).value(this.m_selectedAttribs[j]) - min[j]) * ratio[j]);
					if (inst.instance(i).isMissing(this.m_selectedAttribs[j])) {
						this.m_missing[i][j] = true; // represents missing value
						if (this.m_selectedAttribs[j] == this.m_classIndex) {
							this.m_missing[i][this.m_missing[0].length - 1] = true;
							classIndexProcessed = true;
						}
					}
				}
			}
		}

		if (inst.attribute(this.m_classIndex).isNominal() || inst.attribute(this.m_classIndex).isString()) {
			this.m_type[0] = 1;
			this.m_type[1] = inst.attribute(this.m_classIndex).numValues();
		} else {
			this.m_type[0] = this.m_type[1] = 0;
		}

		if (classIndexProcessed == false) { // class Index has not been processed as
											// class index is not among the selected
											// attribs
			for (int i = 0; i < inst.numInstances(); i++) {
				if (inst.instance(i).isMissing(this.m_classIndex)) {
					this.m_missing[i][this.m_missing[0].length - 1] = true;
				}
			}
		}

		this.m_cp.setColours(this.m_colorList);
	}

	/**
	 * Sets up the UI's attributes lists
	 */
	public void setupAttribLists() {
		String[] tempAttribNames = new String[this.m_data.numAttributes()];
		String type;

		this.m_classAttrib.removeAllItems();
		for (int i = 0; i < tempAttribNames.length; i++) {
			type = " (" + Attribute.typeToStringShort(this.m_data.attribute(i)) + ")";
			tempAttribNames[i] = new String("Colour: " + this.m_data.attribute(i).name() + " " + type);
			this.m_classAttrib.addItem(tempAttribNames[i]);
		}
		if (this.m_data.classIndex() == -1) {
			this.m_classAttrib.setSelectedIndex(tempAttribNames.length - 1);
		} else {
			this.m_classAttrib.setSelectedIndex(this.m_data.classIndex());
		}
		this.m_attribList.setListData(tempAttribNames);
		this.m_attribList.setSelectionInterval(0, tempAttribNames.length - 1);
	}

	/**
	 * Calculates the percentage to resample
	 */
	public void setPercent() {
		if (this.m_data.numInstances() > 700) {
			double percnt = 500D / this.m_data.numInstances() * 100;
			percnt *= 100;
			percnt = Math.round(percnt);
			percnt /= 100;

			this.m_resamplePercent.setText("" + percnt);
		} else {
			this.m_resamplePercent.setText("100");
		}
	}

	/**
	 * This method changes the Instances object of this class to a new one. It
	 * also does all the necessary initializations for displaying the panel. This
	 * must be called before trying to display the panel.
	 *
	 * @param newInst The new set of Instances
	 */
	public void setInstances(final Instances newInst) {

		this.m_osi = null;
		this.m_fastScroll.setSelected(false);
		this.m_data = newInst;
		this.setPercent();
		this.setupAttribLists();
		this.m_rseed.setText("1");
		this.initInternalFields();
		this.m_cp.setInstances(this.m_data);
		this.m_cp.setCindex(this.m_classIndex);
		this.m_updateBt.doClick();
	}

	/**
	 * Main method for testing this class
	 */
	public static void main(final String[] args) {
		final JFrame jf = new JFrame("Weka Explorer: MatrixPanel");
		final JButton setBt = new JButton("Set Instances");
		Instances data = null;
		try {
			if (args.length == 1) {
				data = new Instances(new BufferedReader(new FileReader(args[0])));
			} else {
				System.out.println("Usage: MatrixPanel <arff file>");
				System.exit(-1);
			}
		} catch (IOException ex) {
			ex.printStackTrace();
			System.exit(-1);
		}

		final MatrixPanel mp = new MatrixPanel();
		mp.setInstances(data);
		setBt.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(final ActionEvent e) {
				JFileChooser chooser = new JFileChooser(new java.io.File(System.getProperty("user.dir")));
				ExtensionFileFilter myfilter = new ExtensionFileFilter("arff", "Arff data files");
				chooser.setFileFilter(myfilter);
				int returnVal = chooser.showOpenDialog(jf);

				if (returnVal == JFileChooser.APPROVE_OPTION) {
					try {
						System.out.println("You chose to open this file: " + chooser.getSelectedFile().getName());
						Instances in = new Instances(new FileReader(chooser.getSelectedFile().getAbsolutePath()));
						mp.setInstances(in);
					} catch (Exception ex) {
						ex.printStackTrace();
					}
				}
			}
		});
		// System.out.println("Loaded: "+args[0]+"\nRelation: "+data.relationName()+"\nAttributes: "+data.numAttributes());
		// System.out.println("The attributes are: ");
		// for(int i=0; i<data.numAttributes(); i++)
		// System.out.println(data.attribute(i).name());

		// RepaintManager.currentManager(jf.getRootPane()).setDoubleBufferingEnabled(false);
		jf.getContentPane().setLayout(new BorderLayout());
		jf.getContentPane().add(mp, BorderLayout.CENTER);
		jf.getContentPane().add(setBt, BorderLayout.SOUTH);
		jf.getContentPane().setFont(new java.awt.Font("SansSerif", java.awt.Font.PLAIN, 11));
		jf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		jf.setSize(800, 600);
		jf.setVisible(true);
		jf.repaint();
	}

	/**
	 * Internal class responsible for displaying the actual matrix Requires the
	 * internal data fields of the parent class to be properly initialized before
	 * being created
	 */
	private class Plot extends JPanel implements MouseMotionListener, MouseListener {

		/** for serialization */
		private static final long serialVersionUID = -1721245738439420882L;

		int extpad = 3, intpad = 4, cellSize = 100, cellRange = 100;
		java.awt.Rectangle r;
		java.awt.FontMetrics fm;
		int lastxpos, lastypos;
		JPanel jPlColHeader, jPlRowHeader;

		/**
		 * Constructor
		 */
		public Plot() {
			super();
			this.setToolTipText("blah");
			this.addMouseMotionListener(this);
			this.addMouseListener(this);
			this.initialize();
		}

		/** Initializes the internal fields */
		public void initialize() {
			this.lastxpos = this.lastypos = 0;
			this.cellRange = this.cellSize;
			this.cellSize = this.cellRange + 2 * this.intpad;

			this.jPlColHeader = new JPanel() {
				private static final long serialVersionUID = -9098547751937467506L;
				java.awt.Rectangle r;

				@Override
				public void paint(final Graphics g) {
					this.r = g.getClipBounds();
					g.setColor(this.getBackground());
					g.fillRect(this.r.x, this.r.y, this.r.width, this.r.height);
					g.setFont(MatrixPanel.this.f);
					Plot.this.fm = g.getFontMetrics();
					int xpos = 0, ypos = 0, attribWidth = 0;

					g.setColor(MatrixPanel.this.fontColor);
					xpos = Plot.this.extpad;
					ypos = Plot.this.extpad + Plot.this.fm.getHeight();

					for (int m_selectedAttrib : MatrixPanel.this.m_selectedAttribs) {
						if (xpos + Plot.this.cellSize < this.r.x) {
							xpos += Plot.this.cellSize + Plot.this.extpad;
							continue;
						} else if (xpos > this.r.x + this.r.width) {
							break;
						} else {
							attribWidth = Plot.this.fm.stringWidth(MatrixPanel.this.m_data.attribute(m_selectedAttrib).name());
							g.drawString(MatrixPanel.this.m_data.attribute(m_selectedAttrib).name(), (attribWidth < Plot.this.cellSize) ? (xpos + (Plot.this.cellSize / 2 - attribWidth / 2)) : xpos, ypos);
						}
						xpos += Plot.this.cellSize + Plot.this.extpad;
					}
					Plot.this.fm = null;
					this.r = null;
				}

				@Override
				public Dimension getPreferredSize() {
					Plot.this.fm = this.getFontMetrics(this.getFont());
					return new Dimension(MatrixPanel.this.m_selectedAttribs.length * (Plot.this.cellSize + Plot.this.extpad), 2 * Plot.this.extpad + Plot.this.fm.getHeight());
				}
			};

			this.jPlRowHeader = new JPanel() {
				private static final long serialVersionUID = 8474957069309552844L;

				java.awt.Rectangle r;

				@Override
				public void paint(final Graphics g) {
					this.r = g.getClipBounds();
					g.setColor(this.getBackground());
					g.fillRect(this.r.x, this.r.y, this.r.width, this.r.height);
					g.setFont(MatrixPanel.this.f);
					Plot.this.fm = g.getFontMetrics();
					int xpos = 0, ypos = 0;

					g.setColor(MatrixPanel.this.fontColor);
					xpos = Plot.this.extpad;
					ypos = Plot.this.extpad;

					for (int j = MatrixPanel.this.m_selectedAttribs.length - 1; j >= 0; j--) {
						if (ypos + Plot.this.cellSize < this.r.y) {
							ypos += Plot.this.cellSize + Plot.this.extpad;
							continue;
						} else if (ypos > this.r.y + this.r.height) {
							break;
						} else {
							g.drawString(MatrixPanel.this.m_data.attribute(MatrixPanel.this.m_selectedAttribs[j]).name(), xpos + Plot.this.extpad, ypos + Plot.this.cellSize / 2);
						}
						xpos = Plot.this.extpad;
						ypos += Plot.this.cellSize + Plot.this.extpad;
					}
					this.r = null;
				}

				@Override
				public Dimension getPreferredSize() {
					return new Dimension(100 + Plot.this.extpad, MatrixPanel.this.m_selectedAttribs.length * (Plot.this.cellSize + Plot.this.extpad));
				}
			};
			this.jPlColHeader.setFont(MatrixPanel.this.f);
			this.jPlRowHeader.setFont(MatrixPanel.this.f);
			this.setFont(MatrixPanel.this.f);
		}

		public JPanel getRowHeader() {
			return this.jPlRowHeader;
		}

		public JPanel getColHeader() {
			return this.jPlColHeader;
		}

		@Override
		public void mouseMoved(final MouseEvent e) {
			Graphics g = this.getGraphics();
			int xpos = this.extpad, ypos = this.extpad;

			for (int j = MatrixPanel.this.m_selectedAttribs.length - 1; j >= 0; j--) {
				for (@SuppressWarnings("unused")
				int m_selectedAttrib : MatrixPanel.this.m_selectedAttribs) {
					if (e.getX() >= xpos && e.getX() <= xpos + this.cellSize + this.extpad) {
						if (e.getY() >= ypos && e.getY() <= ypos + this.cellSize + this.extpad) {
							if (xpos != this.lastxpos || ypos != this.lastypos) {
								g.setColor(Color.red);
								g.drawRect(xpos - 1, ypos - 1, this.cellSize + 1, this.cellSize + 1);
								if (this.lastxpos != 0 && this.lastypos != 0) {
									g.setColor(this.getBackground().darker());
									g.drawRect(this.lastxpos - 1, this.lastypos - 1, this.cellSize + 1, this.cellSize + 1);
								}
								this.lastxpos = xpos;
								this.lastypos = ypos;
							}
							return;
						}
					}
					xpos += this.cellSize + this.extpad;
				}
				xpos = this.extpad;
				ypos += this.cellSize + this.extpad;
			}
			if (this.lastxpos != 0 && this.lastypos != 0) {
				g.setColor(this.getBackground().darker());
				g.drawRect(this.lastxpos - 1, this.lastypos - 1, this.cellSize + 1, this.cellSize + 1);
			}
			this.lastxpos = this.lastypos = 0;
		}

		@Override
		public void mouseDragged(final MouseEvent e) {
		}

		@Override
		public void mouseClicked(final MouseEvent e) {
			int i = 0, j = 0, found = 0;

			int xpos = this.extpad, ypos = this.extpad;
			for (j = MatrixPanel.this.m_selectedAttribs.length - 1; j >= 0; j--) {
				for (i = 0; i < MatrixPanel.this.m_selectedAttribs.length; i++) {
					if (e.getX() >= xpos && e.getX() <= xpos + this.cellSize + this.extpad) {
						if (e.getY() >= ypos && e.getY() <= ypos + this.cellSize + this.extpad) {
							found = 1;
							break;
						}
					}
					xpos += this.cellSize + this.extpad;
				}
				if (found == 1) {
					break;
				}
				xpos = this.extpad;
				ypos += this.cellSize + this.extpad;
			}
			if (found == 0) {
				return;
			}

			JFrame jf = Utils.getWekaJFrame("Weka Explorer: Visualizing " + MatrixPanel.this.m_data.relationName(), this);
			VisualizePanel vp = new VisualizePanel();
			try {
				PlotData2D pd = new PlotData2D(MatrixPanel.this.m_data);
				pd.setPlotName("Master Plot");
				vp.setMasterPlot(pd);
				// System.out.println("x: "+i+" y: "+j);
				vp.setXIndex(MatrixPanel.this.m_selectedAttribs[i]);
				vp.setYIndex(MatrixPanel.this.m_selectedAttribs[j]);
				vp.m_ColourCombo.setSelectedIndex(MatrixPanel.this.m_classIndex);
				if (MatrixPanel.this.m_settings != null) {
					vp.applySettings(MatrixPanel.this.m_settings, MatrixPanel.this.m_settingsOwnerID);
				}
			} catch (Exception ex) {
				ex.printStackTrace();
			}
			jf.getContentPane().add(vp);
			jf.pack();
			jf.setSize(800, 600);
			jf.setLocationRelativeTo(SwingUtilities.getWindowAncestor(this));
			jf.setVisible(true);
		}

		@Override
		public void mouseEntered(final MouseEvent e) {
		}

		@Override
		public void mouseExited(final MouseEvent e) {
		}

		@Override
		public void mousePressed(final MouseEvent e) {
		}

		@Override
		public void mouseReleased(final MouseEvent e) {
		}

		/**
		 * sets the new jitter value for the plots
		 */
		public void setJitter(final int newjitter) {
		}

		/**
		 * sets the new size for the plots
		 */
		public void setCellSize(final int newCellSize) {
			this.cellSize = newCellSize;
			this.initialize();
		}

		/**
		 * Returns the X and Y attributes of the plot the mouse is currently on
		 */
		@Override
		public String getToolTipText(final MouseEvent event) {
			int xpos = this.extpad, ypos = this.extpad;

			for (int j = MatrixPanel.this.m_selectedAttribs.length - 1; j >= 0; j--) {
				for (int m_selectedAttrib : MatrixPanel.this.m_selectedAttribs) {
					if (event.getX() >= xpos && event.getX() <= xpos + this.cellSize + this.extpad) {
						if (event.getY() >= ypos && event.getY() <= ypos + this.cellSize + this.extpad) {
							return ("X: " + MatrixPanel.this.m_data.attribute(m_selectedAttrib).name() + " Y: " + MatrixPanel.this.m_data.attribute(MatrixPanel.this.m_selectedAttribs[j]).name() + " (click to enlarge)");
						}
					}
					xpos += this.cellSize + this.extpad;
				}
				xpos = this.extpad;
				ypos += this.cellSize + this.extpad;
			}
			return ("Matrix Panel");
		}

		/**
		 * Paints a single Plot at xpos, ypos. and xattrib and yattrib on X and Y
		 * axes
		 */
		public void paintGraph(final Graphics g, final int xattrib, final int yattrib, final int xpos, final int ypos) {
			int x, y;
			g.setColor(MatrixPanel.this.m_backgroundColor.equals(Color.BLACK) ? MatrixPanel.this.m_backgroundColor.brighter().brighter() : MatrixPanel.this.m_backgroundColor.darker().darker());
			g.drawRect(xpos - 1, ypos - 1, this.cellSize + 1, this.cellSize + 1);
			g.setColor(MatrixPanel.this.m_backgroundColor);
			g.fillRect(xpos, ypos, this.cellSize, this.cellSize);
			for (int i = 0; i < MatrixPanel.this.m_points.length; i++) {

				if (!(MatrixPanel.this.m_missing[i][yattrib] || MatrixPanel.this.m_missing[i][xattrib])) {

					if (MatrixPanel.this.m_type[0] == 0) {
						if (MatrixPanel.this.m_missing[i][MatrixPanel.this.m_missing[0].length - 1]) {
							g.setColor(m_defaultColors[m_defaultColors.length - 1]);
						} else {
							g.setColor(new Color(MatrixPanel.this.m_pointColors[i], 150, (255 - MatrixPanel.this.m_pointColors[i])));
						}
					} else {
						g.setColor(MatrixPanel.this.m_colorList.get(MatrixPanel.this.m_pointColors[i]));
					}

					if (MatrixPanel.this.m_points[i][xattrib] + MatrixPanel.this.jitterVals[i][0] < 0 || MatrixPanel.this.m_points[i][xattrib] + MatrixPanel.this.jitterVals[i][0] > this.cellRange) {
						if (this.cellRange - MatrixPanel.this.m_points[i][yattrib] + MatrixPanel.this.jitterVals[i][1] < 0 || this.cellRange - MatrixPanel.this.m_points[i][yattrib] + MatrixPanel.this.jitterVals[i][1] > this.cellRange) {
							// both x and y out of range don't add jitter
							x = this.intpad + MatrixPanel.this.m_points[i][xattrib];
							y = this.intpad + (this.cellRange - MatrixPanel.this.m_points[i][yattrib]);
						} else {
							// only x out of range
							x = this.intpad + MatrixPanel.this.m_points[i][xattrib];
							y = this.intpad + (this.cellRange - MatrixPanel.this.m_points[i][yattrib]) + MatrixPanel.this.jitterVals[i][1];
						}
					} else if (this.cellRange - MatrixPanel.this.m_points[i][yattrib] + MatrixPanel.this.jitterVals[i][1] < 0 || this.cellRange - MatrixPanel.this.m_points[i][yattrib] + MatrixPanel.this.jitterVals[i][1] > this.cellRange) {
						// only y out of range
						x = this.intpad + MatrixPanel.this.m_points[i][xattrib] + MatrixPanel.this.jitterVals[i][0];
						y = this.intpad + (this.cellRange - MatrixPanel.this.m_points[i][yattrib]);
					} else {
						// none out of range
						x = this.intpad + MatrixPanel.this.m_points[i][xattrib] + MatrixPanel.this.jitterVals[i][0];
						y = this.intpad + (this.cellRange - MatrixPanel.this.m_points[i][yattrib]) + MatrixPanel.this.jitterVals[i][1];
					}
					if (MatrixPanel.this.datapointSize == 1) {
						g.drawLine(x + xpos, y + ypos, x + xpos, y + ypos);
					} else {
						g.drawOval(x + xpos - MatrixPanel.this.datapointSize / 2, y + ypos - MatrixPanel.this.datapointSize / 2, MatrixPanel.this.datapointSize, MatrixPanel.this.datapointSize);
					}
				}
			}
			g.setColor(MatrixPanel.this.fontColor);
		}

		private void createOSI() {
			int iwidth = this.getWidth();
			int iheight = this.getHeight();
			MatrixPanel.this.m_osi = this.createImage(iwidth, iheight);
			this.clearOSI();
		}

		private void clearOSI() {
			if (MatrixPanel.this.m_osi == null) {
				return;
			}

			int iwidth = this.getWidth();
			int iheight = this.getHeight();
			Graphics m = MatrixPanel.this.m_osi.getGraphics();
			m.setColor(this.getBackground().darker().darker());
			m.fillRect(0, 0, iwidth, iheight);
		}

		/**
		 * Paints the matrix of plots in the current visible region
		 */
		public void paintME(final Graphics g) {
			Graphics g2 = g;
			if (MatrixPanel.this.m_osi == null && MatrixPanel.this.m_fastScroll.isSelected()) {
				this.createOSI();
			}
			if (MatrixPanel.this.m_osi != null && MatrixPanel.this.m_fastScroll.isSelected()) {
				g2 = MatrixPanel.this.m_osi.getGraphics();
			}
			this.r = g.getClipBounds();

			g.setColor(this.getBackground());
			g.fillRect(this.r.x, this.r.y, this.r.width, this.r.height);
			g.setColor(MatrixPanel.this.fontColor);

			int xpos = 0, ypos = 0;

			xpos = this.extpad;
			ypos = this.extpad;

			for (int j = MatrixPanel.this.m_selectedAttribs.length - 1; j >= 0; j--) {
				if (ypos + this.cellSize < this.r.y) {
					ypos += this.cellSize + this.extpad;
					continue;
				} else if (ypos > this.r.y + this.r.height) {
					break;
				} else {
					for (int i = 0; i < MatrixPanel.this.m_selectedAttribs.length; i++) {
						if (xpos + this.cellSize < this.r.x) {
							xpos += this.cellSize + this.extpad;
							continue;
						} else if (xpos > this.r.x + this.r.width) {
							break;
						} else if (MatrixPanel.this.m_fastScroll.isSelected()) {
							if (!MatrixPanel.this.m_plottedCells[i][j]) {
								this.paintGraph(g2, i, j, xpos, ypos); // m_selectedAttribs[i],
								// m_selectedAttribs[j], xpos,
								// ypos);
								MatrixPanel.this.m_plottedCells[i][j] = true;
							}
						} else {
							this.paintGraph(g2, i, j, xpos, ypos);
						}
						xpos += this.cellSize + this.extpad;
					}
				}
				xpos = this.extpad;
				ypos += this.cellSize + this.extpad;
			}
		}

		/**
		 * paints this JPanel (PlotsPanel)
		 */
		@Override
		public void paintComponent(final Graphics g) {
			this.paintME(g);
			if (MatrixPanel.this.m_osi != null && MatrixPanel.this.m_fastScroll.isSelected()) {
				g.drawImage(MatrixPanel.this.m_osi, 0, 0, this);
			}
		}
	}

	/**
	 * Set the point size for the plots
	 *
	 * @param pointSize the point size to use
	 */
	public void setPointSize(final int pointSize) {
		if (pointSize <= this.m_pointSize.getMaximum() && pointSize > this.m_pointSize.getMinimum()) {
			this.m_pointSize.setValue(pointSize);
		}
	}

	/**
	 * Set the plot size
	 *
	 * @param plotSize the plot size to use
	 */
	public void setPlotSize(final int plotSize) {
		if (plotSize >= this.m_plotSize.getMinimum() && plotSize <= this.m_plotSize.getMaximum()) {
			this.m_plotSize.setValue(plotSize);
		}
	}

	/**
	 * Set the background colour for the cells in the matrix
	 *
	 * @param c the background colour
	 */
	public void setPlotBackgroundColour(final Color c) {
		this.m_backgroundColor = c;
	}

	/**
	 * @param settings
	 * @param ownerID
	 */
	public void applySettings(final Settings settings, final String ownerID) {
		this.m_settings = settings;
		this.m_settingsOwnerID = ownerID;

		this.setPointSize(settings.getSetting(ownerID, weka.gui.explorer.VisualizePanel.ScatterDefaults.POINT_SIZE_KEY, weka.gui.explorer.VisualizePanel.ScatterDefaults.POINT_SIZE, Environment.getSystemWide()));

		this.setPlotSize(settings.getSetting(ownerID, weka.gui.explorer.VisualizePanel.ScatterDefaults.PLOT_SIZE_KEY, weka.gui.explorer.VisualizePanel.ScatterDefaults.PLOT_SIZE, Environment.getSystemWide()));

		this.setPlotBackgroundColour(settings.getSetting(ownerID, VisualizeUtils.VisualizeDefaults.BACKGROUND_COLOUR_KEY, VisualizeUtils.VisualizeDefaults.BACKGROUND_COLOR, Environment.getSystemWide()));
	}

	/**
	 * Update the display. Typically called after changing plot size, point size
	 * etc.
	 */
	public void updatePanel() {
		// m_selectedAttribs = m_attribList.getSelectedIndices();
		this.initInternalFields();

		Plot a = this.m_plotsPanel;
		a.setCellSize(this.m_plotSize.getValue());
		Dimension d = new Dimension((this.m_selectedAttribs.length) * (a.cellSize + a.extpad) + 2, (this.m_selectedAttribs.length) * (a.cellSize + a.extpad) + 2);
		// System.out.println("Size: "+a.cellSize+" Extpad: "+
		// a.extpad+" selected: "+
		// m_selectedAttribs.length+' '+d);
		a.setPreferredSize(d);
		a.setSize(a.getPreferredSize());
		a.setJitter(this.m_jitter.getValue());

		if (this.m_fastScroll.isSelected() && this.m_clearOSIPlottedCells) {
			this.m_plottedCells = new boolean[this.m_selectedAttribs.length][this.m_selectedAttribs.length];
			this.m_clearOSIPlottedCells = false;
		}

		if (this.m_regenerateOSI) {
			this.m_osi = null;
		}
		this.m_js.revalidate();
		this.m_cp.setColours(this.m_colorList);
		this.m_cp.setCindex(this.m_classIndex);
		this.m_regenerateOSI = false;

		this.repaint();
	}
}
