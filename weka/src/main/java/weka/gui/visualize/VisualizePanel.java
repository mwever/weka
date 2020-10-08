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
 *    VisualizePanel.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.gui.visualize;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.InputEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.BorderFactory;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.SwingConstants;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileFilter;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Settings;
import weka.gui.ExtensionFileFilter;
import weka.gui.Logger;

/**
 * This panel allows the user to visualize a dataset (and if provided) a
 * classifier's/clusterer's predictions in two dimensions.
 *
 * If the user selects a nominal attribute as the colouring attribute then each
 * point is drawn in a colour that corresponds to the discrete value of that
 * attribute for the instance. If the user selects a numeric attribute to colour
 * on, then the points are coloured using a spectrum ranging from blue to red
 * (low values to high).
 *
 * When a classifier's predictions are supplied they are plotted in one of two
 * ways (depending on whether the class is nominal or numeric).<br>
 * For nominal class: an error made by a classifier is plotted as a square in
 * the colour corresponding to the class it predicted.<br>
 * For numeric class: predictions are plotted as varying sized x's, where the
 * size of the x is related to the magnitude of the error.
 *
 * @author Mark Hall (mhall@cs.waikato.ac.nz)
 * @author Malcolm Ware (mfw4@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class VisualizePanel extends PrintablePanel {

	/** for serialization */
	private static final long serialVersionUID = 240108358588153943L;

	/** Inner class to handle plotting */
	protected class PlotPanel extends PrintablePanel implements Plot2DCompanion {

		/** for serialization */
		private static final long serialVersionUID = -4823674171136494204L;

		/** The actual generic plotting panel */
		protected Plot2D m_plot2D = new Plot2D();

		/** The instances from the master plot */
		protected Instances m_plotInstances = null;

		/** The master plot */
		protected PlotData2D m_originalPlot = null;

		/**
		 * Indexes of the attributes to go on the x and y axis and the attribute to
		 * use for colouring and the current shape for drawing
		 */
		protected int m_xIndex = 0;
		protected int m_yIndex = 0;
		protected int m_cIndex = 0;
		protected int m_sIndex = 0;

		/** the offsets of the axes once label metrics are calculated */
		/*
		 * private int m_XaxisStart=0; NOT USED private int m_YaxisStart=0; private
		 * int m_XaxisEnd=0; private int m_YaxisEnd=0;
		 */

		/** True if the user is currently dragging a box. */
		private boolean m_createShape;

		/** contains all the shapes that have been drawn for these attribs */
		private ArrayList<ArrayList<Double>> m_shapes;

		/** contains the points of the shape currently being drawn. */
		private ArrayList<Double> m_shapePoints;

		/** contains the position of the mouse (used for rubberbanding). */
		private final Dimension m_newMousePos;

		/** Constructor */
		public PlotPanel() {
			this.setBackground(this.m_plot2D.getBackground());
			this.setLayout(new BorderLayout());
			this.add(this.m_plot2D, BorderLayout.CENTER);
			this.m_plot2D.setPlotCompanion(this);

			this.m_createShape = false;
			this.m_shapes = null;// //
			this.m_shapePoints = null;
			this.m_newMousePos = new Dimension();

			this.addMouseListener(new MouseAdapter() {
				// /////
				@Override
				public void mousePressed(final MouseEvent e) {
					if ((e.getModifiers() & MouseEvent.BUTTON1_MASK) == MouseEvent.BUTTON1_MASK) {
						//
						if (PlotPanel.this.m_sIndex == 0) {
							// do nothing it will get dealt to in the clicked method
						} else if (PlotPanel.this.m_sIndex == 1) {
							PlotPanel.this.m_createShape = true;
							PlotPanel.this.m_shapePoints = new ArrayList<Double>(5);
							PlotPanel.this.m_shapePoints.add(new Double(PlotPanel.this.m_sIndex));
							PlotPanel.this.m_shapePoints.add(new Double(e.getX()));
							PlotPanel.this.m_shapePoints.add(new Double(e.getY()));
							PlotPanel.this.m_shapePoints.add(new Double(e.getX()));
							PlotPanel.this.m_shapePoints.add(new Double(e.getY()));
							// Graphics g = PlotPanel.this.getGraphics();
							Graphics g = PlotPanel.this.m_plot2D.getGraphics();
							g.setColor(Color.black);
							g.setXORMode(Color.white);
							g.drawRect(PlotPanel.this.m_shapePoints.get(1).intValue(), PlotPanel.this.m_shapePoints.get(2).intValue(), PlotPanel.this.m_shapePoints.get(3).intValue() - PlotPanel.this.m_shapePoints.get(1).intValue(),
									PlotPanel.this.m_shapePoints.get(4).intValue() - PlotPanel.this.m_shapePoints.get(2).intValue());
							g.dispose();
						}
						// System.out.println("clicked");
					}
					// System.out.println("clicked");
				}

				// ////
				@Override
				public void mouseClicked(final MouseEvent e) {

					if ((PlotPanel.this.m_sIndex == 2 || PlotPanel.this.m_sIndex == 3) && (PlotPanel.this.m_createShape || (e.getModifiers() & MouseEvent.BUTTON1_MASK) == MouseEvent.BUTTON1_MASK)) {
						if (PlotPanel.this.m_createShape) {
							// then it has been started already.

							Graphics g = PlotPanel.this.m_plot2D.getGraphics();
							g.setColor(Color.black);
							g.setXORMode(Color.white);
							if ((e.getModifiers() & MouseEvent.BUTTON1_MASK) == MouseEvent.BUTTON1_MASK && !e.isAltDown()) {
								PlotPanel.this.m_shapePoints.add(new Double(PlotPanel.this.m_plot2D.convertToAttribX(e.getX())));

								PlotPanel.this.m_shapePoints.add(new Double(PlotPanel.this.m_plot2D.convertToAttribY(e.getY())));

								PlotPanel.this.m_newMousePos.width = e.getX();
								PlotPanel.this.m_newMousePos.height = e.getY();
								g.drawLine((int) Math.ceil(PlotPanel.this.m_plot2D.convertToPanelX(PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 2).doubleValue())),

										(int) Math.ceil(PlotPanel.this.m_plot2D.convertToPanelY(PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 1).doubleValue())), PlotPanel.this.m_newMousePos.width,
										PlotPanel.this.m_newMousePos.height);

							} else if (PlotPanel.this.m_sIndex == 3) {
								// then extend the lines to infinity
								// (100000 or so should be enough).
								// the area is selected by where the user right clicks
								// the mouse button

								PlotPanel.this.m_createShape = false;
								if (PlotPanel.this.m_shapePoints.size() >= 5) {
									double cx = Math.ceil(PlotPanel.this.m_plot2D.convertToPanelX(PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 4).doubleValue()));

									double cx2 = Math.ceil(PlotPanel.this.m_plot2D.convertToPanelX(PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 2).doubleValue())) - cx;

									cx2 *= 50000;

									double cy = Math.ceil(PlotPanel.this.m_plot2D.convertToPanelY(PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 3).doubleValue()));
									double cy2 = Math.ceil(PlotPanel.this.m_plot2D.convertToPanelY(PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 1).doubleValue())) - cy;
									cy2 *= 50000;

									double cxa = Math.ceil(PlotPanel.this.m_plot2D.convertToPanelX(PlotPanel.this.m_shapePoints.get(3).doubleValue()));
									double cxa2 = Math.ceil(PlotPanel.this.m_plot2D.convertToPanelX(PlotPanel.this.m_shapePoints.get(1).doubleValue())) - cxa;
									cxa2 *= 50000;

									double cya = Math.ceil(PlotPanel.this.m_plot2D.convertToPanelY(PlotPanel.this.m_shapePoints.get(4).doubleValue()));
									double cya2 = Math.ceil(PlotPanel.this.m_plot2D.convertToPanelY(PlotPanel.this.m_shapePoints.get(2).doubleValue())) - cya;

									cya2 *= 50000;

									PlotPanel.this.m_shapePoints.set(1, new Double(PlotPanel.this.m_plot2D.convertToAttribX(cxa2 + cxa)));

									PlotPanel.this.m_shapePoints.set(PlotPanel.this.m_shapePoints.size() - 1, new Double(PlotPanel.this.m_plot2D.convertToAttribY(cy2 + cy)));

									PlotPanel.this.m_shapePoints.set(PlotPanel.this.m_shapePoints.size() - 2, new Double(PlotPanel.this.m_plot2D.convertToAttribX(cx2 + cx)));

									PlotPanel.this.m_shapePoints.set(2, new Double(PlotPanel.this.m_plot2D.convertToAttribY(cya2 + cya)));

									// determine how infinity line should be built

									cy = Double.POSITIVE_INFINITY;
									cy2 = Double.NEGATIVE_INFINITY;
									if (PlotPanel.this.m_shapePoints.get(1).doubleValue() > PlotPanel.this.m_shapePoints.get(3).doubleValue()) {
										if (PlotPanel.this.m_shapePoints.get(2).doubleValue() == PlotPanel.this.m_shapePoints.get(4).doubleValue()) {
											cy = PlotPanel.this.m_shapePoints.get(2).doubleValue();
										}
									}
									if (PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 2).doubleValue() > PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 4).doubleValue()) {
										if (PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 3).doubleValue() == PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 1).doubleValue()) {
											cy2 = PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 1).doubleValue();
										}
									}
									PlotPanel.this.m_shapePoints.add(new Double(cy));
									PlotPanel.this.m_shapePoints.add(new Double(cy2));

									if (!PlotPanel.this.inPolyline(PlotPanel.this.m_shapePoints, PlotPanel.this.m_plot2D.convertToAttribX(e.getX()), PlotPanel.this.m_plot2D.convertToAttribY(e.getY()))) {
										Double tmp = PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 2);
										PlotPanel.this.m_shapePoints.set(PlotPanel.this.m_shapePoints.size() - 2, PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 1));
										PlotPanel.this.m_shapePoints.set(PlotPanel.this.m_shapePoints.size() - 1, tmp);
									}

									if (PlotPanel.this.m_shapes == null) {
										PlotPanel.this.m_shapes = new ArrayList<ArrayList<Double>>(4);
									}
									PlotPanel.this.m_shapes.add(PlotPanel.this.m_shapePoints);

									VisualizePanel.this.m_submit.setText("Submit");
									VisualizePanel.this.m_submit.setActionCommand("Submit");

									VisualizePanel.this.m_submit.setEnabled(true);
								}

								PlotPanel.this.m_shapePoints = null;
								PlotPanel.this.repaint();

							} else {
								// then close the shape
								PlotPanel.this.m_createShape = false;
								if (PlotPanel.this.m_shapePoints.size() >= 7) {
									PlotPanel.this.m_shapePoints.add(PlotPanel.this.m_shapePoints.get(1));
									PlotPanel.this.m_shapePoints.add(PlotPanel.this.m_shapePoints.get(2));
									if (PlotPanel.this.m_shapes == null) {
										PlotPanel.this.m_shapes = new ArrayList<ArrayList<Double>>(4);
									}
									PlotPanel.this.m_shapes.add(PlotPanel.this.m_shapePoints);

									VisualizePanel.this.m_submit.setText("Submit");
									VisualizePanel.this.m_submit.setActionCommand("Submit");

									VisualizePanel.this.m_submit.setEnabled(true);
								}
								PlotPanel.this.m_shapePoints = null;
								PlotPanel.this.repaint();
							}
							g.dispose();
							// repaint();
						} else if ((e.getModifiers() & MouseEvent.BUTTON1_MASK) == MouseEvent.BUTTON1_MASK) {
							// then this is the first point
							PlotPanel.this.m_createShape = true;
							PlotPanel.this.m_shapePoints = new ArrayList<Double>(17);
							PlotPanel.this.m_shapePoints.add(new Double(PlotPanel.this.m_sIndex));
							PlotPanel.this.m_shapePoints.add(new Double(PlotPanel.this.m_plot2D.convertToAttribX(e.getX()))); // the
							// new
							// point
							PlotPanel.this.m_shapePoints.add(new Double(PlotPanel.this.m_plot2D.convertToAttribY(e.getY())));
							PlotPanel.this.m_newMousePos.width = e.getX(); // the temp mouse point
							PlotPanel.this.m_newMousePos.height = e.getY();

							Graphics g = PlotPanel.this.m_plot2D.getGraphics();
							g.setColor(Color.black);
							g.setXORMode(Color.white);
							g.drawLine((int) Math.ceil(PlotPanel.this.m_plot2D.convertToPanelX(PlotPanel.this.m_shapePoints.get(1).doubleValue())),
									(int) Math.ceil(PlotPanel.this.m_plot2D.convertToPanelY(PlotPanel.this.m_shapePoints.get(2).doubleValue())), PlotPanel.this.m_newMousePos.width, PlotPanel.this.m_newMousePos.height);
							g.dispose();
						}
					} else {
						if ((e.getModifiers() & InputEvent.BUTTON1_MASK) == InputEvent.BUTTON1_MASK) {

							PlotPanel.this.m_plot2D.searchPoints(e.getX(), e.getY(), false);
						} else {
							PlotPanel.this.m_plot2D.searchPoints(e.getX(), e.getY(), true);
						}
					}
				}

				// ///////
				@Override
				public void mouseReleased(final MouseEvent e) {

					if (PlotPanel.this.m_createShape) {
						if (PlotPanel.this.m_shapePoints.get(0).intValue() == 1) {
							PlotPanel.this.m_createShape = false;
							Graphics g = PlotPanel.this.m_plot2D.getGraphics();
							g.setColor(Color.black);
							g.setXORMode(Color.white);
							g.drawRect(PlotPanel.this.m_shapePoints.get(1).intValue(), PlotPanel.this.m_shapePoints.get(2).intValue(), PlotPanel.this.m_shapePoints.get(3).intValue() - PlotPanel.this.m_shapePoints.get(1).intValue(),
									PlotPanel.this.m_shapePoints.get(4).intValue() - PlotPanel.this.m_shapePoints.get(2).intValue());

							g.dispose();
							if (PlotPanel.this.checkPoints(PlotPanel.this.m_shapePoints.get(1).doubleValue(), PlotPanel.this.m_shapePoints.get(2).doubleValue())
									&& PlotPanel.this.checkPoints(PlotPanel.this.m_shapePoints.get(3).doubleValue(), PlotPanel.this.m_shapePoints.get(4).doubleValue())) {
								// then the points all land on the screen
								// now do special check for the rectangle
								if (PlotPanel.this.m_shapePoints.get(1).doubleValue() < PlotPanel.this.m_shapePoints.get(3).doubleValue()
										&& PlotPanel.this.m_shapePoints.get(2).doubleValue() < PlotPanel.this.m_shapePoints.get(4).doubleValue()) {
									// then the rectangle is valid
									if (PlotPanel.this.m_shapes == null) {
										PlotPanel.this.m_shapes = new ArrayList<ArrayList<Double>>(2);
									}
									PlotPanel.this.m_shapePoints.set(1, new Double(PlotPanel.this.m_plot2D.convertToAttribX(PlotPanel.this.m_shapePoints.get(1).doubleValue())));
									PlotPanel.this.m_shapePoints.set(2, new Double(PlotPanel.this.m_plot2D.convertToAttribY(PlotPanel.this.m_shapePoints.get(2).doubleValue())));
									PlotPanel.this.m_shapePoints.set(3, new Double(PlotPanel.this.m_plot2D.convertToAttribX(PlotPanel.this.m_shapePoints.get(3).doubleValue())));
									PlotPanel.this.m_shapePoints.set(4, new Double(PlotPanel.this.m_plot2D.convertToAttribY(PlotPanel.this.m_shapePoints.get(4).doubleValue())));

									PlotPanel.this.m_shapes.add(PlotPanel.this.m_shapePoints);

									VisualizePanel.this.m_submit.setText("Submit");
									VisualizePanel.this.m_submit.setActionCommand("Submit");

									VisualizePanel.this.m_submit.setEnabled(true);

									PlotPanel.this.repaint();
								}
							}
							PlotPanel.this.m_shapePoints = null;
						}
					}
				}
			});

			this.addMouseMotionListener(new MouseMotionAdapter() {
				@Override
				public void mouseDragged(final MouseEvent e) {
					// check if the user is dragging a box
					if (PlotPanel.this.m_createShape) {
						if (PlotPanel.this.m_shapePoints.get(0).intValue() == 1) {
							Graphics g = PlotPanel.this.m_plot2D.getGraphics();
							g.setColor(Color.black);
							g.setXORMode(Color.white);
							g.drawRect(PlotPanel.this.m_shapePoints.get(1).intValue(), PlotPanel.this.m_shapePoints.get(2).intValue(), PlotPanel.this.m_shapePoints.get(3).intValue() - PlotPanel.this.m_shapePoints.get(1).intValue(),
									PlotPanel.this.m_shapePoints.get(4).intValue() - PlotPanel.this.m_shapePoints.get(2).intValue());

							PlotPanel.this.m_shapePoints.set(3, new Double(e.getX()));
							PlotPanel.this.m_shapePoints.set(4, new Double(e.getY()));

							g.drawRect(PlotPanel.this.m_shapePoints.get(1).intValue(), PlotPanel.this.m_shapePoints.get(2).intValue(), PlotPanel.this.m_shapePoints.get(3).intValue() - PlotPanel.this.m_shapePoints.get(1).intValue(),
									PlotPanel.this.m_shapePoints.get(4).intValue() - PlotPanel.this.m_shapePoints.get(2).intValue());
							g.dispose();
						}
					}
				}

				@Override
				public void mouseMoved(final MouseEvent e) {
					if (PlotPanel.this.m_createShape) {
						if (PlotPanel.this.m_shapePoints.get(0).intValue() == 2 || PlotPanel.this.m_shapePoints.get(0).intValue() == 3) {
							Graphics g = PlotPanel.this.m_plot2D.getGraphics();
							g.setColor(Color.black);
							g.setXORMode(Color.white);
							g.drawLine((int) Math.ceil(PlotPanel.this.m_plot2D.convertToPanelX(PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 2).doubleValue())),
									(int) Math.ceil(PlotPanel.this.m_plot2D.convertToPanelY(PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 1).doubleValue())), PlotPanel.this.m_newMousePos.width,
									PlotPanel.this.m_newMousePos.height);

							PlotPanel.this.m_newMousePos.width = e.getX();
							PlotPanel.this.m_newMousePos.height = e.getY();

							g.drawLine((int) Math.ceil(PlotPanel.this.m_plot2D.convertToPanelX(PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 2).doubleValue())),
									(int) Math.ceil(PlotPanel.this.m_plot2D.convertToPanelY(PlotPanel.this.m_shapePoints.get(PlotPanel.this.m_shapePoints.size() - 1).doubleValue())), PlotPanel.this.m_newMousePos.width,
									PlotPanel.this.m_newMousePos.height);
							g.dispose();
						}
					}
				}
			});

			VisualizePanel.this.m_submit.addActionListener(new ActionListener() {
				@Override
				public void actionPerformed(final ActionEvent e) {

					if (e.getActionCommand().equals("Submit")) {
						if (VisualizePanel.this.m_splitListener != null && PlotPanel.this.m_shapes != null) {
							// then send the split to the listener
							Instances sub_set1 = new Instances(PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances, 500);
							Instances sub_set2 = new Instances(PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances, 500);

							if (PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances != null) {

								for (int noa = 0; noa < PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances.numInstances(); noa++) {
									if (!PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances.instance(noa).isMissing(PlotPanel.this.m_xIndex)
											&& !PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances.instance(noa).isMissing(PlotPanel.this.m_yIndex)) {

										if (PlotPanel.this.inSplit(PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances.instance(noa))) {
											sub_set1.add(PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances.instance(noa));
										} else {
											sub_set2.add(PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances.instance(noa));
										}
									}
								}
								ArrayList<ArrayList<Double>> tmp = PlotPanel.this.m_shapes;
								PlotPanel.this.cancelShapes();
								VisualizePanel.this.m_splitListener.userDataEvent(new VisualizePanelEvent(tmp, sub_set1, sub_set2, PlotPanel.this.m_xIndex, PlotPanel.this.m_yIndex));
							}
						} else if (PlotPanel.this.m_shapes != null && PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances != null) {
							Instances sub_set1 = new Instances(PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances, 500);
							int count = 0;
							for (int noa = 0; noa < PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances.numInstances(); noa++) {
								if (PlotPanel.this.inSplit(PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances.instance(noa))) {
									sub_set1.add(PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances.instance(noa));
									count++;
								}

							}

							int[] nSizes = null;
							int[] nTypes = null;
							boolean[] connect = null;
							int x = PlotPanel.this.m_xIndex;
							int y = PlotPanel.this.m_yIndex;

							if (PlotPanel.this.m_originalPlot == null) {
								// this sets these instances as the instances
								// to go back to.
								PlotPanel.this.m_originalPlot = PlotPanel.this.m_plot2D.getMasterPlot();
							}

							if (count > 0) {
								nTypes = new int[count];
								nSizes = new int[count];
								connect = new boolean[count];
								count = 0;
								for (int noa = 0; noa < PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances.numInstances(); noa++) {
									if (PlotPanel.this.inSplit(PlotPanel.this.m_plot2D.getMasterPlot().m_plotInstances.instance(noa))) {

										nTypes[count] = PlotPanel.this.m_plot2D.getMasterPlot().m_shapeType[noa];
										nSizes[count] = PlotPanel.this.m_plot2D.getMasterPlot().m_shapeSize[noa];
										connect[count] = PlotPanel.this.m_plot2D.getMasterPlot().m_connectPoints[noa];
										count++;
									}
								}
							}
							PlotPanel.this.cancelShapes();

							PlotData2D newPlot = new PlotData2D(sub_set1);

							try {
								newPlot.setShapeSize(nSizes);
								newPlot.setShapeType(nTypes);
								newPlot.setConnectPoints(connect);

								PlotPanel.this.m_plot2D.removeAllPlots();

								VisualizePanel.this.addPlot(newPlot);
							} catch (Exception ex) {
								System.err.println(ex);
								ex.printStackTrace();
							}

							try {
								VisualizePanel.this.setXIndex(x);
								VisualizePanel.this.setYIndex(y);
							} catch (Exception er) {
								System.out.println("Error : " + er);
								// System.out.println("Part of user input so had to" +
								// " catch here");
							}
						}
					} else if (e.getActionCommand().equals("Reset")) {
						int x = PlotPanel.this.m_xIndex;
						int y = PlotPanel.this.m_yIndex;

						PlotPanel.this.m_plot2D.removeAllPlots();
						try {
							VisualizePanel.this.addPlot(PlotPanel.this.m_originalPlot);
						} catch (Exception ex) {
							System.err.println(ex);
							ex.printStackTrace();
						}

						try {
							VisualizePanel.this.setXIndex(x);
							VisualizePanel.this.setYIndex(y);
						} catch (Exception er) {
							System.out.println("Error : " + er);
						}
					}
				}
			});

			VisualizePanel.this.m_cancel.addActionListener(new ActionListener() {
				@Override
				public void actionPerformed(final ActionEvent e) {
					PlotPanel.this.cancelShapes();
					PlotPanel.this.repaint();
				}
			});
			// //////////
		}

		/**
		 * Apply settings
		 *
		 * @param settings the settings to apply
		 * @param ownerID the ID of the owner perspective, panel etc. This key is
		 *          used when looking up our settings
		 */
		protected void applySettings(final Settings settings, final String ownerID) {
			this.m_plot2D.applySettings(settings, ownerID);
			this.setBackground(this.m_plot2D.getBackground());
			this.repaint();
		}

		/**
		 * Removes all the plots.
		 */
		public void removeAllPlots() {
			this.m_plot2D.removeAllPlots();
			VisualizePanel.this.m_legendPanel.setPlotList(this.m_plot2D.getPlots());
		}

		/**
		 * @return The FastVector containing all the shapes.
		 */
		public ArrayList<ArrayList<Double>> getShapes() {

			return this.m_shapes;
		}

		/**
		 * Sets the list of shapes to empty and also cancels the current shape being
		 * drawn (if applicable).
		 */
		public void cancelShapes() {

			if (VisualizePanel.this.m_splitListener == null) {
				VisualizePanel.this.m_submit.setText("Reset");
				VisualizePanel.this.m_submit.setActionCommand("Reset");

				if (this.m_originalPlot == null || this.m_originalPlot.m_plotInstances == this.m_plotInstances) {
					VisualizePanel.this.m_submit.setEnabled(false);
				} else {
					VisualizePanel.this.m_submit.setEnabled(true);
				}
			} else {
				VisualizePanel.this.m_submit.setEnabled(false);
			}

			this.m_createShape = false;
			this.m_shapePoints = null;
			this.m_shapes = null;
			this.repaint();
		}

		/**
		 * This can be used to set the shapes that should appear.
		 *
		 * @param v The list of shapes.
		 */
		public void setShapes(final ArrayList<ArrayList<Double>> v) {
			// note that this method should be fine for doubles,
			// but anything else that uses something other than doubles
			// (or uneditable objects) could have unsafe copies.
			if (v != null) {
				ArrayList<Double> temp;
				this.m_shapes = new ArrayList<ArrayList<Double>>(v.size());
				for (int noa = 0; noa < v.size(); noa++) {
					temp = new ArrayList<Double>(v.get(noa).size());
					this.m_shapes.add(temp);
					for (int nob = 0; nob < v.get(noa).size(); nob++) {

						temp.add(v.get(noa).get(nob));

					}
				}
			} else {
				this.m_shapes = null;
			}
			this.repaint();
		}

		/**
		 * This will check the values of the screen points passed and make sure that
		 * they land on the screen
		 *
		 * @param x1 The x coord.
		 * @param y1 The y coord.
		 * @return true if the point would land on the screen
		 */
		private boolean checkPoints(final double x1, final double y1) {
			if (x1 < 0 || x1 > this.getSize().width || y1 < 0 || y1 > this.getSize().height) {
				return false;
			}
			return true;
		}

		/**
		 * This will check if an instance is inside or outside of the current
		 * shapes.
		 *
		 * @param i The instance to check.
		 * @return True if 'i' falls inside the shapes, false otherwise.
		 */
		public boolean inSplit(final Instance i) {
			// this will check if the instance lies inside the shapes or not

			if (this.m_shapes != null) {
				ArrayList<Double> stmp;
				double x1, y1, x2, y2;
				for (int noa = 0; noa < this.m_shapes.size(); noa++) {
					stmp = this.m_shapes.get(noa);
					if (stmp.get(0).intValue() == 1) {
						// then rectangle
						x1 = stmp.get(1).doubleValue();
						y1 = stmp.get(2).doubleValue();
						x2 = stmp.get(3).doubleValue();
						y2 = stmp.get(4).doubleValue();
						if (i.value(this.m_xIndex) >= x1 && i.value(this.m_xIndex) <= x2 && i.value(this.m_yIndex) <= y1 && i.value(this.m_yIndex) >= y2) {
							// then is inside split so return true;
							return true;
						}
					} else if (stmp.get(0).intValue() == 2) {
						// then polygon
						if (this.inPoly(stmp, i.value(this.m_xIndex), i.value(this.m_yIndex))) {
							return true;
						}
					} else if (stmp.get(0).intValue() == 3) {
						// then polyline
						if (this.inPolyline(stmp, i.value(this.m_xIndex), i.value(this.m_yIndex))) {
							return true;
						}
					}
				}
			}
			return false;
		}

		/**
		 * Checks to see if the coordinate passed is inside the ployline passed,
		 * Note that this is done using attribute values and not there respective
		 * screen values.
		 *
		 * @param ob The polyline.
		 * @param x The x coord.
		 * @param y The y coord.
		 * @return True if it falls inside the polyline, false otherwise.
		 */
		private boolean inPolyline(final ArrayList<Double> ob, final double x, final double y) {
			// this works similar to the inPoly below except that
			// the first and last lines are treated as extending infinite in one
			// direction and
			// then infinitly in the x dirction their is a line that will
			// normaly be infinite but
			// can be finite in one or both directions

			int countx = 0;
			double vecx, vecy;
			double change;
			double x1, y1, x2, y2;

			for (int noa = 1; noa < ob.size() - 4; noa += 2) {
				y1 = ob.get(noa + 1).doubleValue();
				y2 = ob.get(noa + 3).doubleValue();
				x1 = ob.get(noa).doubleValue();
				x2 = ob.get(noa + 2).doubleValue();

				// System.err.println(y1 + " " + y2 + " " + x1 + " " + x2);
				vecy = y2 - y1;
				vecx = x2 - x1;
				if (noa == 1 && noa == ob.size() - 6) {
					// then do special test first and last edge
					if (vecy != 0) {
						change = (y - y1) / vecy;
						if (vecx * change + x1 >= x) {
							// then intersection
							countx++;
						}
					}
				} else if (noa == 1) {
					if ((y < y2 && vecy > 0) || (y > y2 && vecy < 0)) {
						// now just determine intersection or not
						change = (y - y1) / vecy;
						if (vecx * change + x1 >= x) {
							// then intersection on horiz
							countx++;
						}
					}
				} else if (noa == ob.size() - 6) {
					// then do special test on last edge
					if ((y <= y1 && vecy < 0) || (y >= y1 && vecy > 0)) {
						change = (y - y1) / vecy;
						if (vecx * change + x1 >= x) {
							countx++;
						}
					}
				} else if ((y1 <= y && y < y2) || (y2 < y && y <= y1)) {
					// then continue tests.
					if (vecy == 0) {
						// then lines are parallel stop tests in
						// ofcourse it should never make it this far
					} else {
						change = (y - y1) / vecy;
						if (vecx * change + x1 >= x) {
							// then intersects on horiz
							countx++;
						}
					}
				}
			}

			// now check for intersection with the infinity line
			y1 = ob.get(ob.size() - 2).doubleValue();
			y2 = ob.get(ob.size() - 1).doubleValue();

			if (y1 > y2) {
				// then normal line
				if (y1 >= y && y > y2) {
					countx++;
				}
			} else {
				// then the line segment is inverted
				if (y1 >= y || y > y2) {
					countx++;
				}
			}

			if ((countx % 2) == 1) {
				return true;
			} else {
				return false;
			}
		}

		/**
		 * This checks to see if The coordinate passed is inside the polygon that
		 * was passed.
		 *
		 * @param ob The polygon.
		 * @param x The x coord.
		 * @param y The y coord.
		 * @return True if the coordinate is in the polygon, false otherwise.
		 */
		private boolean inPoly(final ArrayList<Double> ob, final double x, final double y) {
			// brief on how this works
			// it draws a line horizontally from the point to the right (infinitly)
			// it then sees how many lines of the polygon intersect this,
			// if it is even then the point is
			// outside the polygon if it's odd then it's inside the polygon
			int count = 0;
			double vecx, vecy;
			double change;
			double x1, y1, x2, y2;
			for (int noa = 1; noa < ob.size() - 2; noa += 2) {
				y1 = ob.get(noa + 1).doubleValue();
				y2 = ob.get(noa + 3).doubleValue();
				if ((y1 <= y && y < y2) || (y2 < y && y <= y1)) {
					// then continue tests.
					vecy = y2 - y1;
					if (vecy == 0) {
						// then lines are parallel stop tests for this line
					} else {
						x1 = ob.get(noa).doubleValue();
						x2 = ob.get(noa + 2).doubleValue();
						vecx = x2 - x1;
						change = (y - y1) / vecy;
						if (vecx * change + x1 >= x) {
							// then add to count as an intersected line
							count++;
						}
					}
				}
			}
			if ((count % 2) == 1) {
				// then lies inside polygon
				// System.out.println("in");
				return true;
			} else {
				// System.out.println("out");
				return false;
			}
			// System.out.println("WHAT?!?!?!?!!?!??!?!");
			// return false;
		}

		/**
		 * Set level of jitter and repaint the plot using the new jitter value
		 *
		 * @param j the level of jitter
		 */
		public void setJitter(final int j) {
			this.m_plot2D.setJitter(j);
		}

		/**
		 * Set the index of the attribute to go on the x axis
		 *
		 * @param x the index of the attribute to use on the x axis
		 */
		public void setXindex(final int x) {

			// this just ensures that the shapes get disposed of
			// if the attribs change
			if (x != this.m_xIndex) {
				this.cancelShapes();
			}
			this.m_xIndex = x;
			this.m_plot2D.setXindex(x);
			if (VisualizePanel.this.m_showAttBars) {
				VisualizePanel.this.m_attrib.setX(x);
			}
			// this.repaint();
		}

		/**
		 * Set the index of the attribute to go on the y axis
		 *
		 * @param y the index of the attribute to use on the y axis
		 */
		public void setYindex(final int y) {

			// this just ensures that the shapes get disposed of
			// if the attribs change
			if (y != this.m_yIndex) {
				this.cancelShapes();
			}
			this.m_yIndex = y;
			this.m_plot2D.setYindex(y);
			if (VisualizePanel.this.m_showAttBars) {
				VisualizePanel.this.m_attrib.setY(y);
			}
			// this.repaint();
		}

		/**
		 * Set the index of the attribute to use for colouring
		 *
		 * @param c the index of the attribute to use for colouring
		 */
		public void setCindex(final int c) {
			this.m_cIndex = c;
			this.m_plot2D.setCindex(c);
			if (VisualizePanel.this.m_showAttBars) {
				VisualizePanel.this.m_attrib.setCindex(c, this.m_plot2D.getMaxC(), this.m_plot2D.getMinC());
			}
			VisualizePanel.this.m_classPanel.setCindex(c);
			this.repaint();
		}

		/**
		 * Set the index of the attribute to use for the shape.
		 *
		 * @param s the index of the attribute to use for the shape
		 */
		public void setSindex(final int s) {
			if (s != this.m_sIndex) {
				this.m_shapePoints = null;
				this.m_createShape = false;
			}
			this.m_sIndex = s;
			this.repaint();
		}

		/**
		 * Clears all existing plots and sets a new master plot
		 *
		 * @param newPlot the new master plot
		 * @exception Exception if plot could not be added
		 */
		public void setMasterPlot(final PlotData2D newPlot) throws Exception {
			this.m_plot2D.removeAllPlots();
			this.addPlot(newPlot);
		}

		/**
		 * Adds a plot. If there are no plots so far this plot becomes the master
		 * plot and, if it has a custom colour defined then the class panel is
		 * removed.
		 *
		 * @param newPlot the plot to add.
		 * @exception Exception if plot could not be added
		 */
		public void addPlot(final PlotData2D newPlot) throws Exception {
			if (this.m_plot2D.getPlots().size() == 0) {
				this.m_plot2D.addPlot(newPlot);
				if (VisualizePanel.this.m_plotSurround.getComponentCount() > 1 && VisualizePanel.this.m_plotSurround.getComponent(1) == VisualizePanel.this.m_attrib && VisualizePanel.this.m_showAttBars) {
					try {
						VisualizePanel.this.m_attrib.setInstances(newPlot.m_plotInstances);
						VisualizePanel.this.m_attrib.setCindex(0);
						VisualizePanel.this.m_attrib.setX(0);
						VisualizePanel.this.m_attrib.setY(0);
					} catch (Exception ex) {
						// more attributes than the panel can handle?
						// Due to hard coded constraints in GridBagLayout
						VisualizePanel.this.m_plotSurround.remove(VisualizePanel.this.m_attrib);
						System.err.println("Warning : data contains more attributes " + "than can be displayed as attribute bars.");
						if (VisualizePanel.this.m_Log != null) {
							VisualizePanel.this.m_Log.logMessage("Warning : data contains more attributes " + "than can be displayed as attribute bars.");
						}
					}
				} else if (VisualizePanel.this.m_showAttBars) {
					try {
						VisualizePanel.this.m_attrib.setInstances(newPlot.m_plotInstances);
						VisualizePanel.this.m_attrib.setCindex(0);
						VisualizePanel.this.m_attrib.setX(0);
						VisualizePanel.this.m_attrib.setY(0);
						GridBagConstraints constraints = new GridBagConstraints();
						constraints.fill = GridBagConstraints.BOTH;
						constraints.insets = new Insets(0, 0, 0, 0);
						constraints.gridx = 4;
						constraints.gridy = 0;
						constraints.weightx = 1;
						constraints.gridwidth = 1;
						constraints.gridheight = 1;
						constraints.weighty = 5;
						VisualizePanel.this.m_plotSurround.add(VisualizePanel.this.m_attrib, constraints);
					} catch (Exception ex) {
						System.err.println("Warning : data contains more attributes " + "than can be displayed as attribute bars.");
						if (VisualizePanel.this.m_Log != null) {
							VisualizePanel.this.m_Log.logMessage("Warning : data contains more attributes " + "than can be displayed as attribute bars.");
						}
					}
				}
				VisualizePanel.this.m_classPanel.setInstances(newPlot.m_plotInstances);

				this.plotReset(newPlot.m_plotInstances, newPlot.getCindex());
				if (newPlot.m_useCustomColour && VisualizePanel.this.m_showClassPanel) {
					VisualizePanel.this.remove(VisualizePanel.this.m_classSurround);
					this.switchToLegend();
					VisualizePanel.this.m_legendPanel.setPlotList(this.m_plot2D.getPlots());
					VisualizePanel.this.m_ColourCombo.setEnabled(false);
				}
			} else {
				if (!newPlot.m_useCustomColour && VisualizePanel.this.m_showClassPanel) {
					VisualizePanel.this.add(VisualizePanel.this.m_classSurround, BorderLayout.SOUTH);
					VisualizePanel.this.m_ColourCombo.setEnabled(true);
				}
				if (this.m_plot2D.getPlots().size() == 1) {
					this.switchToLegend();
				}
				this.m_plot2D.addPlot(newPlot);
				VisualizePanel.this.m_legendPanel.setPlotList(this.m_plot2D.getPlots());
			}
		}

		/**
		 * Remove the attibute panel and replace it with the legend panel
		 */
		protected void switchToLegend() {

			if (VisualizePanel.this.m_plotSurround.getComponentCount() > 1 && VisualizePanel.this.m_plotSurround.getComponent(1) == VisualizePanel.this.m_attrib) {
				VisualizePanel.this.m_plotSurround.remove(VisualizePanel.this.m_attrib);
			}

			if (VisualizePanel.this.m_plotSurround.getComponentCount() > 1 && VisualizePanel.this.m_plotSurround.getComponent(1) == VisualizePanel.this.m_legendPanel) {
				return;
			}

			GridBagConstraints constraints = new GridBagConstraints();
			constraints.fill = GridBagConstraints.BOTH;
			constraints.insets = new Insets(0, 0, 0, 0);
			constraints.gridx = 4;
			constraints.gridy = 0;
			constraints.weightx = 1;
			constraints.gridwidth = 1;
			constraints.gridheight = 1;
			constraints.weighty = 5;
			VisualizePanel.this.m_plotSurround.add(VisualizePanel.this.m_legendPanel, constraints);
			this.setSindex(0);
			VisualizePanel.this.m_ShapeCombo.setEnabled(false);
		}

		protected void switchToBars() {
			if (VisualizePanel.this.m_plotSurround.getComponentCount() > 1 && VisualizePanel.this.m_plotSurround.getComponent(1) == VisualizePanel.this.m_legendPanel) {
				VisualizePanel.this.m_plotSurround.remove(VisualizePanel.this.m_legendPanel);
			}

			if (VisualizePanel.this.m_plotSurround.getComponentCount() > 1 && VisualizePanel.this.m_plotSurround.getComponent(1) == VisualizePanel.this.m_attrib) {
				return;
			}

			if (VisualizePanel.this.m_showAttBars) {
				try {
					VisualizePanel.this.m_attrib.setInstances(this.m_plot2D.getMasterPlot().m_plotInstances);
					VisualizePanel.this.m_attrib.setCindex(0);
					VisualizePanel.this.m_attrib.setX(0);
					VisualizePanel.this.m_attrib.setY(0);
					GridBagConstraints constraints = new GridBagConstraints();
					constraints.fill = GridBagConstraints.BOTH;
					constraints.insets = new Insets(0, 0, 0, 0);
					constraints.gridx = 4;
					constraints.gridy = 0;
					constraints.weightx = 1;
					constraints.gridwidth = 1;
					constraints.gridheight = 1;
					constraints.weighty = 5;
					VisualizePanel.this.m_plotSurround.add(VisualizePanel.this.m_attrib, constraints);
				} catch (Exception ex) {
					System.err.println("Warning : data contains more attributes " + "than can be displayed as attribute bars.");
					if (VisualizePanel.this.m_Log != null) {
						VisualizePanel.this.m_Log.logMessage("Warning : data contains more attributes " + "than can be displayed as attribute bars.");
					}
				}
			}
		}

		/**
		 * Reset the visualize panel's buttons and the plot panels instances
		 *
		 * @param inst the data
		 * @param cIndex the color index
		 * @throws InterruptedException
		 */
		private void plotReset(final Instances inst, final int cIndex) {
			if (VisualizePanel.this.m_splitListener == null) {
				VisualizePanel.this.m_submit.setText("Reset");
				VisualizePanel.this.m_submit.setActionCommand("Reset");
				// if (m_origInstances == null || m_origInstances == inst) {
				if (this.m_originalPlot == null || this.m_originalPlot.m_plotInstances == inst) {
					VisualizePanel.this.m_submit.setEnabled(false);
				} else {
					VisualizePanel.this.m_submit.setEnabled(true);
				}
			} else {
				VisualizePanel.this.m_submit.setEnabled(false);
			}

			this.m_plotInstances = inst;
			if (VisualizePanel.this.m_splitListener != null) {
				try {
					this.m_plotInstances.randomize(new Random());
				} catch (InterruptedException e) {
					throw new IllegalStateException(e);
				}
			}
			this.m_xIndex = 0;
			this.m_yIndex = 0;
			this.m_cIndex = cIndex;
			this.cancelShapes();
		}

		/**
		 * Set a list of colours to use for plotting points
		 *
		 * @param cols a list of java.awt.Colors
		 */
		public void setColours(final ArrayList<Color> cols) {
			this.m_plot2D.setColours(cols);
			VisualizePanel.this.m_colorList = cols;
		}

		/**
		 * This will draw the shapes created onto the panel. For best visual, this
		 * should be the first thing to be drawn (and it currently is).
		 *
		 * @param gx The graphics context.
		 */
		private void drawShapes(final Graphics gx) {
			// FastVector tmp = m_plot.getShapes();

			if (this.m_shapes != null) {
				ArrayList<Double> stmp;
				int x1, y1, x2, y2;
				for (int noa = 0; noa < this.m_shapes.size(); noa++) {
					stmp = this.m_shapes.get(noa);
					if (stmp.get(0).intValue() == 1) {
						// then rectangle
						x1 = (int) this.m_plot2D.convertToPanelX(stmp.get(1).doubleValue());
						y1 = (int) this.m_plot2D.convertToPanelY(stmp.get(2).doubleValue());
						x2 = (int) this.m_plot2D.convertToPanelX(stmp.get(3).doubleValue());
						y2 = (int) this.m_plot2D.convertToPanelY(stmp.get(4).doubleValue());

						gx.setColor(Color.gray);
						gx.fillRect(x1, y1, x2 - x1, y2 - y1);
						gx.setColor(Color.black);
						gx.drawRect(x1, y1, x2 - x1, y2 - y1);

					} else if (stmp.get(0).intValue() == 2) {
						// then polygon
						int[] ar1, ar2;
						ar1 = this.getXCoords(stmp);
						ar2 = this.getYCoords(stmp);
						gx.setColor(Color.gray);
						gx.fillPolygon(ar1, ar2, (stmp.size() - 1) / 2);
						gx.setColor(Color.black);
						gx.drawPolyline(ar1, ar2, (stmp.size() - 1) / 2);
					} else if (stmp.get(0).intValue() == 3) {
						// then polyline
						int[] ar1, ar2;
						ArrayList<Double> tmp = this.makePolygon(stmp);
						ar1 = this.getXCoords(tmp);
						ar2 = this.getYCoords(tmp);

						gx.setColor(Color.gray);
						gx.fillPolygon(ar1, ar2, (tmp.size() - 1) / 2);
						gx.setColor(Color.black);
						gx.drawPolyline(ar1, ar2, (tmp.size() - 1) / 2);
					}
				}
			}

			if (this.m_shapePoints != null) {
				// then the current image needs to be refreshed
				if (this.m_shapePoints.get(0).intValue() == 2 || this.m_shapePoints.get(0).intValue() == 3) {
					gx.setColor(Color.black);
					gx.setXORMode(Color.white);
					int[] ar1, ar2;
					ar1 = this.getXCoords(this.m_shapePoints);
					ar2 = this.getYCoords(this.m_shapePoints);
					gx.drawPolyline(ar1, ar2, (this.m_shapePoints.size() - 1) / 2);
					this.m_newMousePos.width = (int) Math.ceil(this.m_plot2D.convertToPanelX(this.m_shapePoints.get(this.m_shapePoints.size() - 2).doubleValue()));

					this.m_newMousePos.height = (int) Math.ceil(this.m_plot2D.convertToPanelY(this.m_shapePoints.get(this.m_shapePoints.size() - 1).doubleValue()));

					gx.drawLine((int) Math.ceil(this.m_plot2D.convertToPanelX(this.m_shapePoints.get(this.m_shapePoints.size() - 2).doubleValue())),
							(int) Math.ceil(this.m_plot2D.convertToPanelY(this.m_shapePoints.get(this.m_shapePoints.size() - 1).doubleValue())), this.m_newMousePos.width, this.m_newMousePos.height);
					gx.setPaintMode();
				}
			}
		}

		/**
		 * This is called for polylines to see where there two lines that extend to
		 * infinity cut the border of the view.
		 *
		 * @param x1 an x point along the line
		 * @param y1 the accompanying y point.
		 * @param x2 The x coord of the end point of the line.
		 * @param y2 The y coord of the end point of the line.
		 * @param x 0 or the width of the border line if it has one.
		 * @param y 0 or the height of the border line if it has one.
		 * @param offset The offset for the border line (either for x or y dependant
		 *          on which one doesn't change).
		 * @return double array that contains the coordinate for the point that the
		 *         polyline cuts the border (which ever side that may be).
		 */
		private double[] lineIntersect(final double x1, final double y1, final double x2, final double y2, final double x, final double y, final double offset) {
			// the first 4 params are thestart and end points of a line
			// the next param is either 0 for no change in x or change in x,
			// the next param is the same for y
			// the final 1 is the offset for either x or y (which ever has no change)
			double xval;
			double yval;
			double xn = -100, yn = -100;
			double change;
			if (x == 0) {
				if ((x1 <= offset && offset < x2) || (x1 >= offset && offset > x2)) {
					// then continue
					xval = x1 - x2;
					change = (offset - x2) / xval;
					yn = (y1 - y2) * change + y2;
					if (0 <= yn && yn <= y) {
						// then good
						xn = offset;
					} else {
						// no intersect
						xn = -100;
					}
				}
			} else if (y == 0) {
				if ((y1 <= offset && offset < y2) || (y1 >= offset && offset > y2)) {
					// the continue
					yval = (y1 - y2);
					change = (offset - y2) / yval;
					xn = (x1 - x2) * change + x2;
					if (0 <= xn && xn <= x) {
						// then good
						yn = offset;
					} else {
						xn = -100;
					}
				}
			}
			double[] ret = new double[2];
			ret[0] = xn;
			ret[1] = yn;
			return ret;
		}

		/**
		 * This will convert a polyline to a polygon for drawing purposes So that I
		 * can simply use the polygon drawing function.
		 *
		 * @param v The polyline to convert.
		 * @return A FastVector containing the polygon.
		 */
		private ArrayList<Double> makePolygon(final ArrayList<Double> v) {
			ArrayList<Double> building = new ArrayList<Double>(v.size() + 10);
			double x1, y1, x2, y2;
			int edge1 = 0, edge2 = 0;
			for (int noa = 0; noa < v.size() - 2; noa++) {
				building.add(new Double(v.get(noa).doubleValue()));
			}

			// now clip the lines
			double[] new_coords;
			// note lineIntersect , expects the values to have been converted to
			// screen coords
			// note the first point passed is the one that gets shifted.
			x1 = this.m_plot2D.convertToPanelX(v.get(1).doubleValue());
			y1 = this.m_plot2D.convertToPanelY(v.get(2).doubleValue());
			x2 = this.m_plot2D.convertToPanelX(v.get(3).doubleValue());
			y2 = this.m_plot2D.convertToPanelY(v.get(4).doubleValue());

			if (x1 < 0) {
				// test left
				new_coords = this.lineIntersect(x1, y1, x2, y2, 0, this.getHeight(), 0);
				edge1 = 0;
				if (new_coords[0] < 0) {
					// then not left
					if (y1 < 0) {
						// test top
						new_coords = this.lineIntersect(x1, y1, x2, y2, this.getWidth(), 0, 0);
						edge1 = 1;
					} else {
						// test bottom
						new_coords = this.lineIntersect(x1, y1, x2, y2, this.getWidth(), 0, this.getHeight());
						edge1 = 3;
					}
				}
			} else if (x1 > this.getWidth()) {
				// test right
				new_coords = this.lineIntersect(x1, y1, x2, y2, 0, this.getHeight(), this.getWidth());
				edge1 = 2;
				if (new_coords[0] < 0) {
					// then not right
					if (y1 < 0) {
						// test top
						new_coords = this.lineIntersect(x1, y1, x2, y2, this.getWidth(), 0, 0);
						edge1 = 1;
					} else {
						// test bottom
						new_coords = this.lineIntersect(x1, y1, x2, y2, this.getWidth(), 0, this.getHeight());
						edge1 = 3;
					}
				}
			} else if (y1 < 0) {
				// test top
				new_coords = this.lineIntersect(x1, y1, x2, y2, this.getWidth(), 0, 0);
				edge1 = 1;
			} else {
				// test bottom
				new_coords = this.lineIntersect(x1, y1, x2, y2, this.getWidth(), 0, this.getHeight());
				edge1 = 3;
			}

			building.set(1, new Double(this.m_plot2D.convertToAttribX(new_coords[0])));
			building.set(2, new Double(this.m_plot2D.convertToAttribY(new_coords[1])));

			x1 = this.m_plot2D.convertToPanelX(v.get(v.size() - 4).doubleValue());
			y1 = this.m_plot2D.convertToPanelY(v.get(v.size() - 3).doubleValue());
			x2 = this.m_plot2D.convertToPanelX(v.get(v.size() - 6).doubleValue());
			y2 = this.m_plot2D.convertToPanelY(v.get(v.size() - 5).doubleValue());

			if (x1 < 0) {
				// test left
				new_coords = this.lineIntersect(x1, y1, x2, y2, 0, this.getHeight(), 0);
				edge2 = 0;
				if (new_coords[0] < 0) {
					// then not left
					if (y1 < 0) {
						// test top
						new_coords = this.lineIntersect(x1, y1, x2, y2, this.getWidth(), 0, 0);
						edge2 = 1;
					} else {
						// test bottom
						new_coords = this.lineIntersect(x1, y1, x2, y2, this.getWidth(), 0, this.getHeight());
						edge2 = 3;
					}
				}
			} else if (x1 > this.getWidth()) {
				// test right
				new_coords = this.lineIntersect(x1, y1, x2, y2, 0, this.getHeight(), this.getWidth());
				edge2 = 2;
				if (new_coords[0] < 0) {
					// then not right
					if (y1 < 0) {
						// test top
						new_coords = this.lineIntersect(x1, y1, x2, y2, this.getWidth(), 0, 0);
						edge2 = 1;
					} else {
						// test bottom
						new_coords = this.lineIntersect(x1, y1, x2, y2, this.getWidth(), 0, this.getHeight());
						edge2 = 3;
					}
				}
			} else if (y1 < 0) {
				// test top
				new_coords = this.lineIntersect(x1, y1, x2, y2, this.getWidth(), 0, 0);
				edge2 = 1;
			} else {
				// test bottom
				new_coords = this.lineIntersect(x1, y1, x2, y2, this.getWidth(), 0, this.getHeight());
				edge2 = 3;
			}

			building.set(building.size() - 2, new Double(this.m_plot2D.convertToAttribX(new_coords[0])));
			building.set(building.size() - 1, new Double(this.m_plot2D.convertToAttribY(new_coords[1])));

			// trust me this complicated piece of code will
			// determine what points on the boundary of the view to add to the polygon
			int xp, yp;

			xp = this.getWidth() * ((edge2 & 1) ^ ((edge2 & 2) / 2));
			yp = this.getHeight() * ((edge2 & 2) / 2);
			// System.out.println(((-1 + 4) % 4) + " hoi");

			if (this.inPolyline(v, this.m_plot2D.convertToAttribX(xp), this.m_plot2D.convertToAttribY(yp))) {
				// then add points in a clockwise direction
				building.add(new Double(this.m_plot2D.convertToAttribX(xp)));
				building.add(new Double(this.m_plot2D.convertToAttribY(yp)));
				for (int noa = (edge2 + 1) % 4; noa != edge1; noa = (noa + 1) % 4) {
					xp = this.getWidth() * ((noa & 1) ^ ((noa & 2) / 2));
					yp = this.getHeight() * ((noa & 2) / 2);
					building.add(new Double(this.m_plot2D.convertToAttribX(xp)));
					building.add(new Double(this.m_plot2D.convertToAttribY(yp)));
				}
			} else {
				xp = this.getWidth() * ((edge2 & 2) / 2);
				yp = this.getHeight() * (1 & ~((edge2 & 1) ^ ((edge2 & 2) / 2)));
				if (this.inPolyline(v, this.m_plot2D.convertToAttribX(xp), this.m_plot2D.convertToAttribY(yp))) {
					// then add points in anticlockwise direction
					building.add(new Double(this.m_plot2D.convertToAttribX(xp)));
					building.add(new Double(this.m_plot2D.convertToAttribY(yp)));
					for (int noa = (edge2 + 3) % 4; noa != edge1; noa = (noa + 3) % 4) {
						xp = this.getWidth() * ((noa & 2) / 2);
						yp = this.getHeight() * (1 & ~((noa & 1) ^ ((noa & 2) / 2)));
						building.add(new Double(this.m_plot2D.convertToAttribX(xp)));
						building.add(new Double(this.m_plot2D.convertToAttribY(yp)));
					}
				}
			}
			return building;
		}

		/**
		 * This will extract from a polygon shape its x coodrdinates so that an
		 * awt.Polygon can be created.
		 *
		 * @param v The polygon shape.
		 * @return an int array containing the screen x coords for the polygon.
		 */
		private int[] getXCoords(final ArrayList<Double> v) {
			int cach = (v.size() - 1) / 2;
			int[] ar = new int[cach];
			for (int noa = 0; noa < cach; noa++) {
				ar[noa] = (int) this.m_plot2D.convertToPanelX(v.get(noa * 2 + 1).doubleValue());
			}
			return ar;
		}

		/**
		 * This will extract from a polygon shape its y coordinates so that an
		 * awt.Polygon can be created.
		 *
		 * @param v The polygon shape.
		 * @return an int array containing the screen y coords for the polygon.
		 */
		private int[] getYCoords(final ArrayList<Double> v) {
			int cach = (v.size() - 1) / 2;
			int[] ar = new int[cach];
			for (int noa = 0; noa < cach; noa++) {
				ar[noa] = (int) this.m_plot2D.convertToPanelY(v.get(noa * 2 + 2).doubleValue());
			}
			return ar;
		}

		/**
		 * Renders the polygons if necessary
		 *
		 * @param gx the graphics context
		 */
		@Override
		public void prePlot(final Graphics gx) {
			super.paintComponent(gx);
			if (this.m_plotInstances != null) {
				this.drawShapes(gx); // will be in paintComponent of ShapePlot2D
			}
		}
	}

	/** default colours for colouring discrete class */
	protected Color[] m_DefaultColors = { Color.blue, Color.red, Color.green, Color.cyan, Color.pink, new Color(255, 0, 255), Color.orange, new Color(255, 0, 0), new Color(0, 255, 0), Color.white };

	/** Lets the user select the attribute for the x axis */
	protected JComboBox m_XCombo = new JComboBox();

	/** Lets the user select the attribute for the y axis */
	protected JComboBox m_YCombo = new JComboBox();

	/** Lets the user select the attribute to use for colouring */
	protected JComboBox m_ColourCombo = new JComboBox();

	/**
	 * Lets the user select the shape they want to create for instance selection.
	 */
	protected JComboBox m_ShapeCombo = new JComboBox();

	/** Button for the user to enter the splits. */
	protected JButton m_submit = new JButton("Submit");

	/** Button for the user to remove all splits. */
	protected JButton m_cancel = new JButton("Clear");

	/** Button for the user to open the visualized set of instances */
	protected JButton m_openBut = new JButton("Open");

	/** Button for the user to save the visualized set of instances */
	protected JButton m_saveBut = new JButton("Save");

	/** Stop the combos from growing out of control */
	private final Dimension COMBO_SIZE = new Dimension(250, this.m_saveBut.getPreferredSize().height);

	/** file chooser for saving instances */
	protected JFileChooser m_FileChooser = new JFileChooser(new File(System.getProperty("user.dir")));

	/** Filter to ensure only arff files are selected */
	protected FileFilter m_ArffFilter = new ExtensionFileFilter(Instances.FILE_EXTENSION, "Arff data files");

	/** Label for the jitter slider */
	protected JLabel m_JitterLab = new JLabel("Jitter", SwingConstants.RIGHT);

	/** The jitter slider */
	protected JSlider m_Jitter = new JSlider(0, 50, 0);

	/** The panel that displays the plot */
	protected PlotPanel m_plot = new PlotPanel();

	/**
	 * The panel that displays the attributes , using color to represent another
	 * attribute.
	 */
	protected AttributePanel m_attrib = new AttributePanel(this.m_plot.m_plot2D.getBackground());

	/** The panel that displays legend info if there is more than one plot */
	protected LegendPanel m_legendPanel = new LegendPanel();

	/** Panel that surrounds the plot panel with a titled border */
	protected JPanel m_plotSurround = new JPanel();

	/** Panel that surrounds the class panel with a titled border */
	protected JPanel m_classSurround = new JPanel();

	/**
	 * An optional listener that we will inform when ComboBox selections change
	 */
	protected ActionListener listener = null;

	/**
	 * An optional listener that we will inform when the user creates a split to
	 * seperate instances.
	 */
	protected VisualizePanelListener m_splitListener = null;

	/**
	 * The name of the plot (not currently displayed, but can be used in the
	 * containing Frame or Panel)
	 */
	protected String m_plotName = "";

	/** The panel that displays the legend for the colouring attribute */
	protected ClassPanel m_classPanel = new ClassPanel(this.m_plot.m_plot2D.getBackground());

	/** The list of the colors used */
	protected ArrayList<Color> m_colorList;

	/**
	 * These hold the names of preferred columns to visualize on---if the user has
	 * defined them in the Visualize.props file
	 */
	protected String m_preferredXDimension = null;
	protected String m_preferredYDimension = null;
	protected String m_preferredColourDimension = null;

	/** Show the attribute bar panel */
	protected boolean m_showAttBars = true;

	/** Show the class panel **/
	protected boolean m_showClassPanel = true;

	/** the logger */
	protected Logger m_Log;

	/**
	 * Sets the Logger to receive informational messages
	 *
	 * @param newLog the Logger that will now get info messages
	 */
	public void setLog(final Logger newLog) {
		this.m_Log = newLog;
	}

	/**
	 * Set whether the attribute bars should be shown or not. If turned off via
	 * this method then any setting in the properties file (if exists) is ignored.
	 *
	 * @param sab false if attribute bars are not to be displayed.
	 */
	public void setShowAttBars(final boolean sab) {
		if (!sab && this.m_showAttBars) {
			this.m_plotSurround.remove(this.m_attrib);
		} else if (sab && !this.m_showAttBars) {
			GridBagConstraints constraints = new GridBagConstraints();
			constraints.insets = new Insets(0, 0, 0, 0);
			constraints.gridx = 4;
			constraints.gridy = 0;
			constraints.weightx = 1;
			constraints.gridwidth = 1;
			constraints.gridheight = 1;
			constraints.weighty = 5;
			this.m_plotSurround.add(this.m_attrib, constraints);
		}
		this.m_showAttBars = sab;
		this.repaint();
	}

	/**
	 * Gets whether or not attribute bars are being displayed.
	 *
	 * @return true if attribute bars are being displayed.
	 */
	public boolean getShowAttBars() {
		return this.m_showAttBars;
	}

	/**
	 * Set whether the class panel should be shown or not.
	 *
	 * @param scp false if class panel is not to be displayed
	 */
	public void setShowClassPanel(final boolean scp) {
		if (!scp && this.m_showClassPanel) {
			this.remove(this.m_classSurround);
		} else if (scp && !this.m_showClassPanel) {
			this.add(this.m_classSurround, BorderLayout.SOUTH);
		}
		this.m_showClassPanel = scp;
		this.repaint();
	}

	/**
	 * Gets whether or not the class panel is being displayed.
	 *
	 * @return true if the class panel is being displayed.
	 */
	public boolean getShowClassPanel() {
		return this.m_showClassPanel;
	}

	/**
	 * This constructor allows a VisualizePanelListener to be set.
	 *
	 * @param ls the listener to use
	 */
	public VisualizePanel(final VisualizePanelListener ls) {
		this();
		this.m_splitListener = ls;
	}

	/**
	 * Set the properties for the VisualizePanel
	 *
	 * @param relationName the name of the relation, can be null
	 */
	private void setProperties(final String relationName) {
		if (VisualizeUtils.VISUALIZE_PROPERTIES != null) {
			String thisClass = this.getClass().getName();
			if (relationName == null) {

				String showAttBars = thisClass + ".displayAttributeBars";

				String val = VisualizeUtils.VISUALIZE_PROPERTIES.getProperty(showAttBars);
				if (val == null) {
					// System.err.println("Displaying attribute bars ");
					// m_showAttBars = true;
				} else {
					// only check if this hasn't been turned off programatically
					if (this.m_showAttBars) {
						if (val.compareTo("true") == 0 || val.compareTo("on") == 0) {
							// System.err.println("Displaying attribute bars ");
							this.m_showAttBars = true;
						} else {
							this.m_showAttBars = false;
						}
					}
				}
			} else {
				/*
				 * System.err.println("Looking for preferred visualization dimensions for "
				 * +relationName);
				 */
				String xcolKey = thisClass + "." + relationName + ".XDimension";
				String ycolKey = thisClass + "." + relationName + ".YDimension";
				String ccolKey = thisClass + "." + relationName + ".ColourDimension";

				this.m_preferredXDimension = VisualizeUtils.VISUALIZE_PROPERTIES.getProperty(xcolKey);
				/*
				 * if (m_preferredXDimension == null) {
				 * System.err.println("No preferred X dimension found in "
				 * +VisualizeUtils.PROPERTY_FILE +" for "+xcolKey); } else {
				 * System.err.println("Setting preferred X dimension to "
				 * +m_preferredXDimension); }
				 */
				this.m_preferredYDimension = VisualizeUtils.VISUALIZE_PROPERTIES.getProperty(ycolKey);
				/*
				 * if (m_preferredYDimension == null) {
				 * System.err.println("No preferred Y dimension found in "
				 * +VisualizeUtils.PROPERTY_FILE +" for "+ycolKey); } else {
				 * System.err.println("Setting preferred dimension Y to "
				 * +m_preferredYDimension); }
				 */
				this.m_preferredColourDimension = VisualizeUtils.VISUALIZE_PROPERTIES.getProperty(ccolKey);
				/*
				 * if (m_preferredColourDimension == null) {
				 * System.err.println("No preferred Colour dimension found in "
				 * +VisualizeUtils.PROPERTY_FILE +" for "+ycolKey); } else {
				 * System.err.println("Setting preferred Colour dimension to "
				 * +m_preferredColourDimension); }
				 */
			}
		}
	}

	/**
	 * Apply settings
	 *
	 * @param settings the settings to apply
	 * @param ownerID the ID of the owner perspective, panel etc. to use when
	 *          looking up settings
	 */
	public void applySettings(final Settings settings, final String ownerID) {
		this.m_plot.applySettings(settings, ownerID);
		this.m_attrib.applySettings(settings, ownerID);
		this.repaint();
	}

	/**
	 * Constructor
	 */
	public VisualizePanel() {
		super();
		this.setProperties(null);
		this.m_FileChooser.setFileFilter(this.m_ArffFilter);
		this.m_FileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

		this.m_XCombo.setToolTipText("Select the attribute for the x axis");
		this.m_YCombo.setToolTipText("Select the attribute for the y axis");
		this.m_ColourCombo.setToolTipText("Select the attribute to colour on");
		this.m_ShapeCombo.setToolTipText("Select the shape to use for data selection");

		this.m_XCombo.setPreferredSize(this.COMBO_SIZE);
		this.m_YCombo.setPreferredSize(this.COMBO_SIZE);
		this.m_ColourCombo.setPreferredSize(this.COMBO_SIZE);
		this.m_ShapeCombo.setPreferredSize(this.COMBO_SIZE);

		this.m_XCombo.setMaximumSize(this.COMBO_SIZE);
		this.m_YCombo.setMaximumSize(this.COMBO_SIZE);
		this.m_ColourCombo.setMaximumSize(this.COMBO_SIZE);
		this.m_ShapeCombo.setMaximumSize(this.COMBO_SIZE);

		this.m_XCombo.setMinimumSize(this.COMBO_SIZE);
		this.m_YCombo.setMinimumSize(this.COMBO_SIZE);
		this.m_ColourCombo.setMinimumSize(this.COMBO_SIZE);
		this.m_ShapeCombo.setMinimumSize(this.COMBO_SIZE);
		// ////////
		this.m_XCombo.setEnabled(false);
		this.m_YCombo.setEnabled(false);
		this.m_ColourCombo.setEnabled(false);
		this.m_ShapeCombo.setEnabled(false);

		// tell the class panel and the legend panel that we want to know when
		// colours change
		this.m_classPanel.addRepaintNotify(this);
		this.m_legendPanel.addRepaintNotify(this);

		// Check the default colours against the background colour of the
		// plot panel. If any are equal to the background colour then
		// change them (so they are visible :-)
		for (int i = 0; i < this.m_DefaultColors.length; i++) {
			Color c = this.m_DefaultColors[i];
			if (c.equals(this.m_plot.m_plot2D.getBackground())) {
				int red = c.getRed();
				int blue = c.getBlue();
				int green = c.getGreen();
				red += (red < 128) ? (255 - red) / 2 : -(red / 2);
				blue += (blue < 128) ? (blue - red) / 2 : -(blue / 2);
				green += (green < 128) ? (255 - green) / 2 : -(green / 2);
				this.m_DefaultColors[i] = new Color(red, green, blue);
			}
		}
		this.m_classPanel.setDefaultColourList(this.m_DefaultColors);
		this.m_attrib.setDefaultColourList(this.m_DefaultColors);

		this.m_colorList = new ArrayList<Color>(10);
		for (int noa = this.m_colorList.size(); noa < 10; noa++) {
			Color pc = this.m_DefaultColors[noa % 10];
			int ija = noa / 10;
			ija *= 2;
			for (int j = 0; j < ija; j++) {
				pc = pc.darker();
			}

			this.m_colorList.add(pc);
		}
		this.m_plot.setColours(this.m_colorList);
		this.m_classPanel.setColours(this.m_colorList);
		this.m_attrib.setColours(this.m_colorList);
		this.m_attrib.addAttributePanelListener(new AttributePanelListener() {
			@Override
			public void attributeSelectionChange(final AttributePanelEvent e) {
				if (e.m_xChange) {
					VisualizePanel.this.m_XCombo.setSelectedIndex(e.m_indexVal);
				} else {
					VisualizePanel.this.m_YCombo.setSelectedIndex(e.m_indexVal);
				}
			}
		});

		this.m_XCombo.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(final ActionEvent e) {
				int selected = VisualizePanel.this.m_XCombo.getSelectedIndex();
				if (selected < 0) {
					selected = 0;
				}
				VisualizePanel.this.m_plot.setXindex(selected);

				// try sending on the event if anyone is listening
				if (VisualizePanel.this.listener != null) {
					VisualizePanel.this.listener.actionPerformed(e);
				}
			}
		});

		this.m_YCombo.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(final ActionEvent e) {
				int selected = VisualizePanel.this.m_YCombo.getSelectedIndex();
				if (selected < 0) {
					selected = 0;
				}
				VisualizePanel.this.m_plot.setYindex(selected);

				// try sending on the event if anyone is listening
				if (VisualizePanel.this.listener != null) {
					VisualizePanel.this.listener.actionPerformed(e);
				}
			}
		});

		this.m_ColourCombo.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(final ActionEvent e) {
				int selected = VisualizePanel.this.m_ColourCombo.getSelectedIndex();
				if (selected < 0) {
					selected = 0;
				}
				VisualizePanel.this.m_plot.setCindex(selected);

				if (VisualizePanel.this.listener != null) {
					VisualizePanel.this.listener.actionPerformed(e);
				}
			}
		});

		// /////
		this.m_ShapeCombo.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(final ActionEvent e) {
				int selected = VisualizePanel.this.m_ShapeCombo.getSelectedIndex();
				if (selected < 0) {
					selected = 0;
				}
				VisualizePanel.this.m_plot.setSindex(selected);
				// try sending on the event if anyone is listening
				if (VisualizePanel.this.listener != null) {
					VisualizePanel.this.listener.actionPerformed(e);
				}
			}
		});

		// /////////////////////////////////////

		this.m_Jitter.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(final ChangeEvent e) {
				VisualizePanel.this.m_plot.setJitter(VisualizePanel.this.m_Jitter.getValue());
			}
		});

		this.m_openBut.setToolTipText("Loads previously saved instances from a file");
		this.m_openBut.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(final ActionEvent e) {
				VisualizePanel.this.openVisibleInstances();
			}
		});

		this.m_saveBut.setEnabled(false);
		this.m_saveBut.setToolTipText("Save the visible instances to a file");
		this.m_saveBut.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(final ActionEvent e) {
				VisualizePanel.this.saveVisibleInstances();
			}
		});

		JPanel combos = new JPanel();
		GridBagLayout gb = new GridBagLayout();
		GridBagConstraints constraints = new GridBagConstraints();

		this.m_XCombo.setLightWeightPopupEnabled(false);
		this.m_YCombo.setLightWeightPopupEnabled(false);
		this.m_ColourCombo.setLightWeightPopupEnabled(false);
		this.m_ShapeCombo.setLightWeightPopupEnabled(false);
		combos.setBorder(BorderFactory.createEmptyBorder(10, 5, 10, 5));

		combos.setLayout(gb);
		constraints.gridx = 0;
		constraints.gridy = 0;
		constraints.weightx = 5;
		constraints.fill = GridBagConstraints.HORIZONTAL;
		constraints.gridwidth = 2;
		constraints.gridheight = 1;
		constraints.insets = new Insets(0, 2, 0, 2);
		combos.add(this.m_XCombo, constraints);
		constraints.gridx = 2;
		constraints.gridy = 0;
		constraints.weightx = 5;
		constraints.gridwidth = 2;
		constraints.gridheight = 1;
		combos.add(this.m_YCombo, constraints);
		constraints.gridx = 0;
		constraints.gridy = 1;
		constraints.weightx = 5;
		constraints.gridwidth = 2;
		constraints.gridheight = 1;
		combos.add(this.m_ColourCombo, constraints);
		//
		constraints.gridx = 2;
		constraints.gridy = 1;
		constraints.weightx = 5;
		constraints.gridwidth = 2;
		constraints.gridheight = 1;
		combos.add(this.m_ShapeCombo, constraints);

		JPanel mbts = new JPanel();
		mbts.setLayout(new GridLayout(1, 4));
		mbts.add(this.m_submit);
		mbts.add(this.m_cancel);
		mbts.add(this.m_openBut);
		mbts.add(this.m_saveBut);

		constraints.gridx = 0;
		constraints.gridy = 2;
		constraints.weightx = 5;
		constraints.gridwidth = 2;
		constraints.gridheight = 1;
		combos.add(mbts, constraints);

		// //////////////////////////////
		constraints.gridx = 2;
		constraints.gridy = 2;
		constraints.weightx = 5;
		constraints.gridwidth = 1;
		constraints.gridheight = 1;
		constraints.insets = new Insets(10, 0, 0, 5);
		combos.add(this.m_JitterLab, constraints);
		constraints.gridx = 3;
		constraints.gridy = 2;
		constraints.weightx = 5;
		constraints.insets = new Insets(10, 0, 0, 0);
		combos.add(this.m_Jitter, constraints);

		this.m_classSurround = new JPanel();
		this.m_classSurround.setBorder(BorderFactory.createTitledBorder("Class colour"));
		this.m_classSurround.setLayout(new BorderLayout());

		this.m_classPanel.setBorder(BorderFactory.createEmptyBorder(15, 10, 10, 10));
		this.m_classSurround.add(this.m_classPanel, BorderLayout.CENTER);

		GridBagLayout gb2 = new GridBagLayout();
		this.m_plotSurround.setBorder(BorderFactory.createTitledBorder("Plot"));
		this.m_plotSurround.setLayout(gb2);

		constraints.fill = GridBagConstraints.BOTH;
		constraints.insets = new Insets(0, 0, 0, 10);
		constraints.gridx = 0;
		constraints.gridy = 0;
		constraints.weightx = 3;
		constraints.gridwidth = 4;
		constraints.gridheight = 1;
		constraints.weighty = 5;
		this.m_plotSurround.add(this.m_plot, constraints);

		if (this.m_showAttBars) {
			constraints.insets = new Insets(0, 0, 0, 0);
			constraints.gridx = 4;
			constraints.gridy = 0;
			constraints.weightx = 1;
			constraints.gridwidth = 1;
			constraints.gridheight = 1;
			constraints.weighty = 5;
			this.m_plotSurround.add(this.m_attrib, constraints);
		}

		this.setLayout(new BorderLayout());
		this.add(combos, BorderLayout.NORTH);
		this.add(this.m_plotSurround, BorderLayout.CENTER);
		this.add(this.m_classSurround, BorderLayout.SOUTH);

		String[] SNames = new String[4];
		SNames[0] = "Select Instance";
		SNames[1] = "Rectangle";
		SNames[2] = "Polygon";
		SNames[3] = "Polyline";

		this.m_ShapeCombo.setModel(new DefaultComboBoxModel(SNames));
		this.m_ShapeCombo.setEnabled(true);
	}

	/**
	 * displays the previously saved instances
	 *
	 * @param insts the instances to display
	 * @throws Exception if display is not possible
	 */
	protected void openVisibleInstances(final Instances insts) throws Exception {
		PlotData2D tempd = new PlotData2D(insts);
		tempd.setPlotName(insts.relationName());
		tempd.addInstanceNumberAttribute();
		this.m_plot.m_plot2D.removeAllPlots();
		this.addPlot(tempd);

		// modify title
		Component parent = this.getParent();
		while (parent != null) {
			if (parent instanceof JFrame) {
				((JFrame) parent).setTitle("Weka Classifier Visualize: " + insts.relationName() + " (display only)");
				break;
			} else {
				parent = parent.getParent();
			}
		}
	}

	/**
	 * Loads previously saved instances from a file
	 */
	protected void openVisibleInstances() {
		try {
			int returnVal = this.m_FileChooser.showOpenDialog(this);
			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File sFile = this.m_FileChooser.getSelectedFile();
				if (!sFile.getName().toLowerCase().endsWith(Instances.FILE_EXTENSION)) {
					sFile = new File(sFile.getParent(), sFile.getName() + Instances.FILE_EXTENSION);
				}
				File selected = sFile;
				Instances insts = new Instances(new BufferedReader(new FileReader(selected)));
				this.openVisibleInstances(insts);
			}
		} catch (Exception ex) {
			ex.printStackTrace();
			this.m_plot.m_plot2D.removeAllPlots();
			JOptionPane.showMessageDialog(this, ex.getMessage(), "Error loading file...", JOptionPane.ERROR_MESSAGE);
		}
	}

	/**
	 * Save the currently visible set of instances to a file
	 */
	private void saveVisibleInstances() {
		ArrayList<PlotData2D> plots = this.m_plot.m_plot2D.getPlots();
		if (plots != null) {
			PlotData2D master = plots.get(0);
			Instances saveInsts = new Instances(master.getPlotInstances());
			for (int i = 1; i < plots.size(); i++) {
				PlotData2D temp = plots.get(i);
				Instances addInsts = temp.getPlotInstances();
				for (int j = 0; j < addInsts.numInstances(); j++) {
					saveInsts.add(addInsts.instance(j));
				}
			}
			try {
				int returnVal = this.m_FileChooser.showSaveDialog(this);
				if (returnVal == JFileChooser.APPROVE_OPTION) {
					File sFile = this.m_FileChooser.getSelectedFile();
					if (!sFile.getName().toLowerCase().endsWith(Instances.FILE_EXTENSION)) {
						sFile = new File(sFile.getParent(), sFile.getName() + Instances.FILE_EXTENSION);
					}
					File selected = sFile;
					Writer w = new BufferedWriter(new FileWriter(selected));
					w.write(saveInsts.toString());
					w.close();
				}
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}
	}

	/**
	 * Set the index for colouring.
	 *
	 * @param index the index of the attribute to use for colouring
	 * @param enableCombo false if the colouring combo box should be disabled
	 */
	public void setColourIndex(final int index, final boolean enableCombo) {
		if (index >= 0) {
			this.m_ColourCombo.setSelectedIndex(index);
		} else {
			this.m_ColourCombo.setSelectedIndex(0);
		}
		this.m_ColourCombo.setEnabled(enableCombo);
	}

	/**
	 * Sets the index used for colouring. If this method is called then the
	 * supplied index is used and the combo box for selecting colouring attribute
	 * is disabled.
	 *
	 * @param index the index of the attribute to use for colouring
	 */
	public void setColourIndex(final int index) {
		this.setColourIndex(index, false);
	}

	/**
	 * Set the index of the attribute for the x axis
	 *
	 * @param index the index for the x axis
	 * @exception Exception if index is out of range.
	 */
	public void setXIndex(final int index) throws Exception {
		if (index >= 0 && index < this.m_XCombo.getItemCount()) {
			this.m_XCombo.setSelectedIndex(index);
		} else {
			throw new Exception("x index is out of range!");
		}
	}

	/**
	 * Get the index of the attribute on the x axis
	 *
	 * @return the index of the attribute on the x axis
	 */
	public int getXIndex() {
		return this.m_XCombo.getSelectedIndex();
	}

	/**
	 * Set the index of the attribute for the y axis
	 *
	 * @param index the index for the y axis
	 * @exception Exception if index is out of range.
	 */
	public void setYIndex(final int index) throws Exception {
		if (index >= 0 && index < this.m_YCombo.getItemCount()) {
			this.m_YCombo.setSelectedIndex(index);
		} else {
			throw new Exception("y index is out of range!");
		}
	}

	/**
	 * Get the index of the attribute on the y axis
	 *
	 * @return the index of the attribute on the x axis
	 */
	public int getYIndex() {
		return this.m_YCombo.getSelectedIndex();
	}

	/**
	 * Get the index of the attribute selected for coloring
	 *
	 * @return the index of the attribute on the x axis
	 */
	public int getCIndex() {
		return this.m_ColourCombo.getSelectedIndex();
	}

	/**
	 * Get the index of the shape selected for creating splits.
	 *
	 * @return The index of the shape.
	 */
	public int getSIndex() {
		return this.m_ShapeCombo.getSelectedIndex();
	}

	/**
	 * Set the shape for creating splits.
	 *
	 * @param index The index of the shape.
	 * @exception Exception if index is out of range.
	 */
	public void setSIndex(final int index) throws Exception {
		if (index >= 0 && index < this.m_ShapeCombo.getItemCount()) {
			this.m_ShapeCombo.setSelectedIndex(index);
		} else {
			throw new Exception("s index is out of range!");
		}
	}

	/**
	 * Add a listener for this visualize panel
	 *
	 * @param act an ActionListener
	 */
	public void addActionListener(final ActionListener act) {
		this.listener = act;
	}

	/**
	 * Set a name for this plot
	 *
	 * @param plotName the name for the plot
	 */
	@Override
	public void setName(final String plotName) {
		this.m_plotName = plotName;
	}

	/**
	 * Returns the name associated with this plot. "" is returned if no name is
	 * set.
	 *
	 * @return the name of the plot
	 */
	@Override
	public String getName() {
		return this.m_plotName;
	}

	/**
	 * Get the master plot's instances
	 *
	 * @return the master plot's instances
	 */
	public Instances getInstances() {
		return this.m_plot.m_plotInstances;
	}

	/**
	 * Sets the Colors in use for a different attrib if it is not a nominal attrib
	 * and or does not have more possible values then this will do nothing.
	 * otherwise it will add default colors to see that there is a color for the
	 * attrib to begin with.
	 *
	 * @param a The index of the attribute to color.
	 * @param i The instances object that contains the attribute.
	 */
	protected void newColorAttribute(final int a, final Instances i) {
		if (i.attribute(a).isNominal()) {
			for (int noa = this.m_colorList.size(); noa < i.attribute(a).numValues(); noa++) {
				Color pc = this.m_DefaultColors[noa % 10];
				int ija = noa / 10;
				ija *= 2;
				for (int j = 0; j < ija; j++) {
					pc = pc.brighter();
				}

				this.m_colorList.add(pc);
			}
			this.m_plot.setColours(this.m_colorList);
			this.m_attrib.setColours(this.m_colorList);
			this.m_classPanel.setColours(this.m_colorList);
		}
	}

	/**
	 * This will set the shapes for the instances.
	 *
	 * @param l A list of the shapes, providing that the objects in the lists are
	 *          non editable the data will be kept intact.
	 */
	public void setShapes(final ArrayList<ArrayList<Double>> l) {
		this.m_plot.setShapes(l);
	}

	/**
	 * Tells the panel to use a new set of instances.
	 *
	 * @param inst a set of Instances
	 */
	public void setInstances(final Instances inst) {
		if (inst.numAttributes() > 0 && inst.numInstances() > 0) {
			this.newColorAttribute(inst.numAttributes() - 1, inst);
		}

		PlotData2D temp = new PlotData2D(inst);
		temp.setPlotName(inst.relationName());

		try {
			this.setMasterPlot(temp);
		} catch (Exception ex) {
			System.err.println(ex);
			ex.printStackTrace();
		}
	}

	/**
	 * initializes the comboboxes based on the data
	 *
	 * @param inst the data to base the combobox-setup on
	 */
	public void setUpComboBoxes(final Instances inst) {
		this.setProperties(inst.relationName());
		int prefX = -1;
		int prefY = -1;
		if (inst.numAttributes() > 1) {
			prefY = 1;
		}
		int prefC = -1;
		String[] XNames = new String[inst.numAttributes()];
		String[] YNames = new String[inst.numAttributes()];
		String[] CNames = new String[inst.numAttributes()];
		for (int i = 0; i < XNames.length; i++) {
			String type = " (" + Attribute.typeToStringShort(inst.attribute(i)) + ")";
			XNames[i] = "X: " + inst.attribute(i).name() + type;
			YNames[i] = "Y: " + inst.attribute(i).name() + type;
			CNames[i] = "Colour: " + inst.attribute(i).name() + type;
			if (this.m_preferredXDimension != null) {
				if (this.m_preferredXDimension.compareTo(inst.attribute(i).name()) == 0) {
					prefX = i;
					// System.err.println("Found preferred X dimension");
				}
			}
			if (this.m_preferredYDimension != null) {
				if (this.m_preferredYDimension.compareTo(inst.attribute(i).name()) == 0) {
					prefY = i;
					// System.err.println("Found preferred Y dimension");
				}
			}
			if (this.m_preferredColourDimension != null) {
				if (this.m_preferredColourDimension.compareTo(inst.attribute(i).name()) == 0) {
					prefC = i;
					// System.err.println("Found preferred Colour dimension");
				}
			}
		}
		this.m_XCombo.setModel(new DefaultComboBoxModel(XNames));
		this.m_YCombo.setModel(new DefaultComboBoxModel(YNames));

		this.m_ColourCombo.setModel(new DefaultComboBoxModel(CNames));
		// m_ShapeCombo.setModel(new DefaultComboBoxModel(SNames));
		// m_ShapeCombo.setEnabled(true);
		this.m_XCombo.setEnabled(true);
		this.m_YCombo.setEnabled(true);

		if (this.m_splitListener == null) {
			this.m_ColourCombo.setEnabled(true);
			this.m_ColourCombo.setSelectedIndex(inst.numAttributes() - 1);
		}
		this.m_plotSurround.setBorder((BorderFactory.createTitledBorder("Plot: " + inst.relationName())));
		try {
			if (prefX != -1) {
				this.setXIndex(prefX);
			}
			if (prefY != -1) {
				this.setYIndex(prefY);
			}
			if (prefC != -1) {
				this.m_ColourCombo.setSelectedIndex(prefC);
			}
		} catch (Exception ex) {
			System.err.println("Problem setting preferred Visualization dimensions");
		}
	}

	/**
	 * Removes all the plots.
	 */
	public void removeAllPlots() {
		this.m_plot.removeAllPlots();
	}

	/**
	 * Set the master plot for the visualize panel
	 *
	 * @param newPlot the new master plot
	 * @exception Exception if the master plot could not be set
	 */
	public void setMasterPlot(final PlotData2D newPlot) throws Exception {
		this.m_plot.setMasterPlot(newPlot);
		this.setUpComboBoxes(newPlot.m_plotInstances);
		this.m_saveBut.setEnabled(true);
		this.repaint();
	}

	/**
	 * Set a new plot to the visualize panel
	 *
	 * @param newPlot the new plot to add
	 * @exception Exception if the plot could not be added
	 */
	public void addPlot(final PlotData2D newPlot) throws Exception {
		this.m_plot.addPlot(newPlot);
		if (this.m_plot.m_plot2D.getMasterPlot() != null) {
			this.setUpComboBoxes(newPlot.m_plotInstances);
		}
		this.m_saveBut.setEnabled(true);
		this.repaint();
	}

	/**
	 * Returns the underlying plot panel.
	 *
	 * @return the plot panel
	 */
	public PlotPanel getPlotPanel() {
		return this.m_plot;
	}

	/**
	 * Main method for testing this class
	 *
	 * @param args the commandline parameters
	 */
	public static void main(final String[] args) {
		try {
			if (args.length < 1) {
				System.err.println("Usage : weka.gui.visualize.VisualizePanel " + "<dataset> [<dataset> <dataset>...]");
				System.exit(1);
			}

			weka.core.logging.Logger.log(weka.core.logging.Logger.Level.INFO, "Logging started");
			final javax.swing.JFrame jf = new javax.swing.JFrame("Weka Explorer: Visualize");
			jf.setSize(500, 400);
			jf.getContentPane().setLayout(new BorderLayout());
			final VisualizePanel sp = new VisualizePanel();

			jf.getContentPane().add(sp, BorderLayout.CENTER);
			jf.addWindowListener(new java.awt.event.WindowAdapter() {
				@Override
				public void windowClosing(final java.awt.event.WindowEvent e) {
					jf.dispose();
					System.exit(0);
				}
			});

			jf.setVisible(true);
			if (args.length >= 1) {
				for (int j = 0; j < args.length; j++) {
					System.err.println("Loading instances from " + args[j]);
					java.io.Reader r = new java.io.BufferedReader(new java.io.FileReader(args[j]));
					Instances i = new Instances(r);
					i.setClassIndex(i.numAttributes() - 1);
					PlotData2D pd1 = new PlotData2D(i);

					if (j == 0) {
						pd1.setPlotName("Master plot");
						sp.setMasterPlot(pd1);
					} else {
						pd1.setPlotName("Plot " + (j + 1));
						pd1.m_useCustomColour = true;
						pd1.m_customColour = (j % 2 == 0) ? Color.red : Color.blue;
						sp.addPlot(pd1);
					}
				}
			}
		} catch (Exception ex) {
			ex.printStackTrace();
			System.err.println(ex.getMessage());
		}
	}
}
