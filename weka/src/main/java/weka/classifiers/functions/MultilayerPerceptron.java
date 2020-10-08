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
 *    MultilayerPerceptron.java
 *    Copyright (C) 2000-2012 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.functions;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.IterativeClassifier;
import weka.classifiers.functions.neural.LinearUnit;
import weka.classifiers.functions.neural.NeuralConnection;
import weka.classifiers.functions.neural.NeuralNode;
import weka.classifiers.functions.neural.SigmoidUnit;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

/**
 * <!-- globalinfo-start --> A Classifier that uses backpropagation to classify instances.<br/>
 * This network can be built by hand, created by an algorithm or both. The network can also be
 * monitored and modified during training time. The nodes in this network are all sigmoid (except
 * for when the class is numeric in which case the the output nodes become unthresholded linear
 * units).
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -L &lt;learning rate&gt;
 *  Learning Rate for the backpropagation algorithm.
 *  (Value should be between 0 - 1, Default = 0.3).
 * </pre>
 *
 * <pre>
 * -M &lt;momentum&gt;
 *  Momentum Rate for the backpropagation algorithm.
 *  (Value should be between 0 - 1, Default = 0.2).
 * </pre>
 *
 * <pre>
 * -N &lt;number of epochs&gt;
 *  Number of epochs to train through.
 *  (Default = 500).
 * </pre>
 *
 * <pre>
 * -V &lt;percentage size of validation set&gt;
 *  Percentage size of validation set to use to terminate
 *  training (if this is non zero it can pre-empt num of epochs.
 *  (Value should be between 0 - 100, Default = 0).
 * </pre>
 *
 * <pre>
 * -S &lt;seed&gt;
 *  The value used to seed the random number generator
 *  (Value should be &gt;= 0 and and a long, Default = 0).
 * </pre>
 *
 * <pre>
 * -E &lt;threshold for number of consequetive errors&gt;
 *  The consequetive number of errors allowed for validation
 *  testing before the netwrok terminates.
 *  (Value should be &gt; 0, Default = 20).
 * </pre>
 *
 * <pre>
 * -G
 *  GUI will be opened.
 *  (Use this to bring up a GUI).
 * </pre>
 *
 * <pre>
 * -A
 *  Autocreation of the network connections will NOT be done.
 *  (This will be ignored if -G is NOT set)
 * </pre>
 *
 * <pre>
 * -B
 *  A NominalToBinary filter will NOT automatically be used.
 *  (Set this to not use a NominalToBinary filter).
 * </pre>
 *
 * <pre>
 * -H &lt;comma seperated numbers for nodes on each layer&gt;
 *  The hidden layers to be created for the network.
 *  (Value should be a list of comma separated Natural
 *  numbers or the letters 'a' = (attribs + classes) / 2,
 *  'i' = attribs, 'o' = classes, 't' = attribs .+ classes)
 *  for wildcard values, Default = a).
 * </pre>
 *
 * <pre>
 * -C
 *  Normalizing a numeric class will NOT be done.
 *  (Set this to not normalize the class if it's numeric).
 * </pre>
 *
 * <pre>
 * -I
 *  Normalizing the attributes will NOT be done.
 *  (Set this to not normalize the attributes).
 * </pre>
 *
 * <pre>
 * -R
 *  Reseting the network will NOT be allowed.
 *  (Set this to not allow the network to reset).
 * </pre>
 *
 * <pre>
 * -D
 *  Learning rate decay will occur.
 *  (Set this to cause the learning rate to decay).
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Malcolm Ware (mfw4@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class MultilayerPerceptron extends AbstractClassifier implements OptionHandler, WeightedInstancesHandler, Randomizable, IterativeClassifier {

	/** for serialization */
	private static final long serialVersionUID = -5990607817048210779L;

	/**
	 * Main method for testing this class.
	 *
	 * @param argv
	 *          should contain command line options (see setOptions)
	 */
	public static void main(final String[] argv) {
		runClassifier(new MultilayerPerceptron(), argv);
	}

	/**
	 * This inner class is used to connect the nodes in the network up to the data that they are
	 * classifying, Note that objects of this class are only suitable to go on the attribute side or
	 * class side of the network and not both.
	 */
	protected class NeuralEnd extends NeuralConnection {

		/** for serialization */
		static final long serialVersionUID = 7305185603191183338L;

		/**
		 * the value that represents the instance value this node represents. For an input it is the
		 * attribute number, for an output, if nominal it is the class value.
		 */
		private int m_link;

		/** True if node is an input, False if it's an output. */
		private boolean m_input;

		/**
		 * Constructor
		 */
		public NeuralEnd(final String id) {
			super(id);

			this.m_link = 0;
			this.m_input = true;

		}

		/**
		 * Call this function to determine if the point at x,y is on the unit.
		 *
		 * @param g
		 *          The graphics context for font size info.
		 * @param x
		 *          The x coord.
		 * @param y
		 *          The y coord.
		 * @param w
		 *          The width of the display.
		 * @param h
		 *          The height of the display.
		 * @return True if the point is on the unit, false otherwise.
		 */
		@Override
		public boolean onUnit(final Graphics g, final int x, final int y, final int w, final int h) {

			FontMetrics fm = g.getFontMetrics();
			int l = (int) (this.m_x * w) - fm.stringWidth(this.m_id) / 2;
			int t = (int) (this.m_y * h) - fm.getHeight() / 2;
			if (x < l || x > l + fm.stringWidth(this.m_id) + 4 || y < t || y > t + fm.getHeight() + fm.getDescent() + 4) {
				return false;
			}
			return true;

		}

		/**
		 * This will draw the node id to the graphics context.
		 *
		 * @param g
		 *          The graphics context.
		 * @param w
		 *          The width of the drawing area.
		 * @param h
		 *          The height of the drawing area.
		 */
		@Override
		public void drawNode(final Graphics g, final int w, final int h) {

			if ((this.m_type & PURE_INPUT) == PURE_INPUT) {
				g.setColor(Color.green);
			} else {
				g.setColor(Color.orange);
			}

			FontMetrics fm = g.getFontMetrics();
			int l = (int) (this.m_x * w) - fm.stringWidth(this.m_id) / 2;
			int t = (int) (this.m_y * h) - fm.getHeight() / 2;
			g.fill3DRect(l, t, fm.stringWidth(this.m_id) + 4, fm.getHeight() + fm.getDescent() + 4, true);
			g.setColor(Color.black);

			g.drawString(this.m_id, l + 2, t + fm.getHeight() + 2);

		}

		/**
		 * Call this function to draw the node highlighted.
		 *
		 * @param g
		 *          The graphics context.
		 * @param w
		 *          The width of the drawing area.
		 * @param h
		 *          The height of the drawing area.
		 */
		@Override
		public void drawHighlight(final Graphics g, final int w, final int h) {

			g.setColor(Color.black);
			FontMetrics fm = g.getFontMetrics();
			int l = (int) (this.m_x * w) - fm.stringWidth(this.m_id) / 2;
			int t = (int) (this.m_y * h) - fm.getHeight() / 2;
			g.fillRect(l - 2, t - 2, fm.stringWidth(this.m_id) + 8, fm.getHeight() + fm.getDescent() + 8);
			this.drawNode(g, w, h);
		}

		/**
		 * Call this to get the output value of this unit.
		 *
		 * @param calculate
		 *          True if the value should be calculated if it hasn't been already.
		 * @return The output value, or NaN, if the value has not been calculated.
		 */
		@Override
		public double outputValue(final boolean calculate) {

			if (Double.isNaN(this.m_unitValue) && calculate) {
				if (this.m_input) {
					if (MultilayerPerceptron.this.m_currentInstance.isMissing(this.m_link)) {
						this.m_unitValue = 0;
					} else {

						this.m_unitValue = MultilayerPerceptron.this.m_currentInstance.value(this.m_link);
					}
				} else {
					// node is an output.
					this.m_unitValue = 0;
					for (int noa = 0; noa < this.m_numInputs; noa++) {
						this.m_unitValue += this.m_inputList[noa].outputValue(true);

					}
					if (MultilayerPerceptron.this.m_numeric && MultilayerPerceptron.this.m_normalizeClass) {
						// then scale the value;
						// this scales linearly from between -1 and 1
						this.m_unitValue = this.m_unitValue * MultilayerPerceptron.this.m_attributeRanges[MultilayerPerceptron.this.m_instances.classIndex()]
								+ MultilayerPerceptron.this.m_attributeBases[MultilayerPerceptron.this.m_instances.classIndex()];
					}
				}
			}
			return this.m_unitValue;

		}

		/**
		 * Call this to get the error value of this unit, which in this case is the difference between the
		 * predicted class, and the actual class.
		 *
		 * @param calculate
		 *          True if the value should be calculated if it hasn't been already.
		 * @return The error value, or NaN, if the value has not been calculated.
		 */
		@Override
		public double errorValue(final boolean calculate) {

			if (!Double.isNaN(this.m_unitValue) && Double.isNaN(this.m_unitError) && calculate) {

				if (this.m_input) {
					this.m_unitError = 0;
					for (int noa = 0; noa < this.m_numOutputs; noa++) {
						this.m_unitError += this.m_outputList[noa].errorValue(true);
					}
				} else {
					if (MultilayerPerceptron.this.m_currentInstance.classIsMissing()) {
						this.m_unitError = .1;
					} else if (MultilayerPerceptron.this.m_instances.classAttribute().isNominal()) {
						if (MultilayerPerceptron.this.m_currentInstance.classValue() == this.m_link) {
							this.m_unitError = 1 - this.m_unitValue;
						} else {
							this.m_unitError = 0 - this.m_unitValue;
						}
					} else if (MultilayerPerceptron.this.m_numeric) {

						if (MultilayerPerceptron.this.m_normalizeClass) {
							if (MultilayerPerceptron.this.m_attributeRanges[MultilayerPerceptron.this.m_instances.classIndex()] == 0) {
								this.m_unitError = 0;
							} else {
								this.m_unitError = (MultilayerPerceptron.this.m_currentInstance.classValue() - this.m_unitValue) / MultilayerPerceptron.this.m_attributeRanges[MultilayerPerceptron.this.m_instances.classIndex()];
								// m_numericRange;

							}
						} else {
							this.m_unitError = MultilayerPerceptron.this.m_currentInstance.classValue() - this.m_unitValue;
						}
					}
				}
			}
			return this.m_unitError;
		}

		/**
		 * Call this to reset the value and error for this unit, ready for the next run. This will also call
		 * the reset function of all units that are connected as inputs to this one. This is also the time
		 * that the update for the listeners will be performed.
		 */
		@Override
		public void reset() {

			if (!Double.isNaN(this.m_unitValue) || !Double.isNaN(this.m_unitError)) {
				this.m_unitValue = Double.NaN;
				this.m_unitError = Double.NaN;
				this.m_weightsUpdated = false;
				for (int noa = 0; noa < this.m_numInputs; noa++) {
					this.m_inputList[noa].reset();
				}
			}
		}

		/**
		 * Call this to have the connection save the current weights.
		 */
		@Override
		public void saveWeights() {
			for (int i = 0; i < this.m_numInputs; i++) {
				this.m_inputList[i].saveWeights();
			}
		}

		/**
		 * Call this to have the connection restore from the saved weights.
		 */
		@Override
		public void restoreWeights() {
			for (int i = 0; i < this.m_numInputs; i++) {
				this.m_inputList[i].restoreWeights();
			}
		}

		/**
		 * Call this function to set What this end unit represents.
		 *
		 * @param input
		 *          True if this unit is used for entering an attribute, False if it's used for determining
		 *          a class value.
		 * @param val
		 *          The attribute number or class type that this unit represents. (for nominal attributes).
		 */
		public void setLink(final boolean input, final int val) throws Exception {
			this.m_input = input;

			if (input) {
				this.m_type = PURE_INPUT;
			} else {
				this.m_type = PURE_OUTPUT;
			}
			if (val < 0 || (input && val > MultilayerPerceptron.this.m_instances.numAttributes())
					|| (!input && MultilayerPerceptron.this.m_instances.classAttribute().isNominal() && val > MultilayerPerceptron.this.m_instances.classAttribute().numValues())) {
				this.m_link = 0;
			} else {
				this.m_link = val;
			}
		}

		/**
		 * @return link for this node.
		 */
		public int getLink() {
			return this.m_link;
		}

		/**
		 * Returns the revision string.
		 *
		 * @return the revision
		 */
		@Override
		public String getRevision() {
			return RevisionUtils.extract("$Revision$");
		}
	}

	/**
	 * Inner class used to draw the nodes onto.(uses the node lists!!) This will also handle the user
	 * input.
	 */
	private class NodePanel extends JPanel implements RevisionHandler {

		/** for serialization */
		static final long serialVersionUID = -3067621833388149984L;

		/**
		 * The constructor.
		 */
		public NodePanel() {

			this.addMouseListener(new MouseAdapter() {

				@Override
				public void mousePressed(final MouseEvent e) {

					if (!MultilayerPerceptron.this.m_stopped) {
						return;
					}
					if ((e.getModifiers() & MouseEvent.BUTTON1_MASK) == MouseEvent.BUTTON1_MASK && !e.isAltDown()) {
						Graphics g = NodePanel.this.getGraphics();
						int x = e.getX();
						int y = e.getY();
						int w = NodePanel.this.getWidth();
						int h = NodePanel.this.getHeight();
						ArrayList<NeuralConnection> tmp = new ArrayList<>(4);
						for (int noa = 0; noa < MultilayerPerceptron.this.m_numAttributes; noa++) {
							if (MultilayerPerceptron.this.m_inputs[noa].onUnit(g, x, y, w, h)) {
								tmp.add(MultilayerPerceptron.this.m_inputs[noa]);
								NodePanel.this.selection(tmp, (e.getModifiers() & MouseEvent.CTRL_MASK) == MouseEvent.CTRL_MASK, true);
								return;
							}
						}
						for (int noa = 0; noa < MultilayerPerceptron.this.m_numClasses; noa++) {
							if (MultilayerPerceptron.this.m_outputs[noa].onUnit(g, x, y, w, h)) {
								tmp.add(MultilayerPerceptron.this.m_outputs[noa]);
								NodePanel.this.selection(tmp, (e.getModifiers() & MouseEvent.CTRL_MASK) == MouseEvent.CTRL_MASK, true);
								return;
							}
						}
						for (NeuralConnection m_neuralNode : MultilayerPerceptron.this.m_neuralNodes) {
							if (m_neuralNode.onUnit(g, x, y, w, h)) {
								tmp.add(m_neuralNode);
								NodePanel.this.selection(tmp, (e.getModifiers() & MouseEvent.CTRL_MASK) == MouseEvent.CTRL_MASK, true);
								return;
							}

						}
						NeuralNode temp = new NeuralNode(String.valueOf(MultilayerPerceptron.this.m_nextId), MultilayerPerceptron.this.m_random, MultilayerPerceptron.this.m_sigmoidUnit);
						MultilayerPerceptron.this.m_nextId++;
						temp.setX((double) e.getX() / w);
						temp.setY((double) e.getY() / h);
						tmp.add(temp);
						MultilayerPerceptron.this.addNode(temp);
						NodePanel.this.selection(tmp, (e.getModifiers() & MouseEvent.CTRL_MASK) == MouseEvent.CTRL_MASK, true);
					} else {
						// then right click
						Graphics g = NodePanel.this.getGraphics();
						int x = e.getX();
						int y = e.getY();
						int w = NodePanel.this.getWidth();
						int h = NodePanel.this.getHeight();
						ArrayList<NeuralConnection> tmp = new ArrayList<>(4);
						for (int noa = 0; noa < MultilayerPerceptron.this.m_numAttributes; noa++) {
							if (MultilayerPerceptron.this.m_inputs[noa].onUnit(g, x, y, w, h)) {
								tmp.add(MultilayerPerceptron.this.m_inputs[noa]);
								NodePanel.this.selection(tmp, (e.getModifiers() & MouseEvent.CTRL_MASK) == MouseEvent.CTRL_MASK, false);
								return;
							}

						}
						for (int noa = 0; noa < MultilayerPerceptron.this.m_numClasses; noa++) {
							if (MultilayerPerceptron.this.m_outputs[noa].onUnit(g, x, y, w, h)) {
								tmp.add(MultilayerPerceptron.this.m_outputs[noa]);
								NodePanel.this.selection(tmp, (e.getModifiers() & MouseEvent.CTRL_MASK) == MouseEvent.CTRL_MASK, false);
								return;
							}
						}
						for (NeuralConnection m_neuralNode : MultilayerPerceptron.this.m_neuralNodes) {
							if (m_neuralNode.onUnit(g, x, y, w, h)) {
								tmp.add(m_neuralNode);
								NodePanel.this.selection(tmp, (e.getModifiers() & MouseEvent.CTRL_MASK) == MouseEvent.CTRL_MASK, false);
								return;
							}
						}
						NodePanel.this.selection(null, (e.getModifiers() & MouseEvent.CTRL_MASK) == MouseEvent.CTRL_MASK, false);
					}
				}
			});
		}

		/**
		 * This function gets called when the user has clicked something It will amend the current selection
		 * or connect the current selection to the new selection. Or if nothing was selected and the right
		 * button was used it will delete the node.
		 *
		 * @param v
		 *          The units that were selected.
		 * @param ctrl
		 *          True if ctrl was held down.
		 * @param left
		 *          True if it was the left mouse button.
		 */
		private void selection(final ArrayList<NeuralConnection> v, final boolean ctrl, final boolean left) {

			if (v == null) {
				// then unselect all.
				MultilayerPerceptron.this.m_selected.clear();
				this.repaint();
				return;
			}

			// then exclusive or the new selection with the current one.
			if ((ctrl || MultilayerPerceptron.this.m_selected.size() == 0) && left) {
				boolean removed = false;
				for (int noa = 0; noa < v.size(); noa++) {
					removed = false;
					for (int nob = 0; nob < MultilayerPerceptron.this.m_selected.size(); nob++) {
						if (v.get(noa) == MultilayerPerceptron.this.m_selected.get(nob)) {
							// then remove that element
							MultilayerPerceptron.this.m_selected.remove(nob);
							removed = true;
							break;
						}
					}
					if (!removed) {
						MultilayerPerceptron.this.m_selected.add(v.get(noa));
					}
				}
				this.repaint();
				return;
			}

			if (left) {
				// then connect the current selection to the new one.
				for (int noa = 0; noa < MultilayerPerceptron.this.m_selected.size(); noa++) {
					for (int nob = 0; nob < v.size(); nob++) {
						NeuralConnection.connect(MultilayerPerceptron.this.m_selected.get(noa), v.get(nob));
					}
				}
			} else if (MultilayerPerceptron.this.m_selected.size() > 0) {
				// then disconnect the current selection from the new one.

				for (int noa = 0; noa < MultilayerPerceptron.this.m_selected.size(); noa++) {
					for (int nob = 0; nob < v.size(); nob++) {
						NeuralConnection.disconnect(MultilayerPerceptron.this.m_selected.get(noa), v.get(nob));

						NeuralConnection.disconnect(v.get(nob), MultilayerPerceptron.this.m_selected.get(noa));

					}
				}
			} else {
				// then remove the selected node. (it was right clicked while
				// no other units were selected
				for (int noa = 0; noa < v.size(); noa++) {
					v.get(noa).removeAllInputs();
					v.get(noa).removeAllOutputs();
					MultilayerPerceptron.this.removeNode(v.get(noa));
				}
			}
			this.repaint();
		}

		/**
		 * This will paint the nodes ontot the panel.
		 *
		 * @param g
		 *          The graphics context.
		 */
		@Override
		public void paintComponent(final Graphics g) {

			super.paintComponent(g);
			int x = this.getWidth();
			int y = this.getHeight();
			if (25 * MultilayerPerceptron.this.m_numAttributes > 25 * MultilayerPerceptron.this.m_numClasses && 25 * MultilayerPerceptron.this.m_numAttributes > y) {
				this.setSize(x, 25 * MultilayerPerceptron.this.m_numAttributes);
			} else if (25 * MultilayerPerceptron.this.m_numClasses > y) {
				this.setSize(x, 25 * MultilayerPerceptron.this.m_numClasses);
			} else {
				this.setSize(x, y);
			}

			y = this.getHeight();
			for (int noa = 0; noa < MultilayerPerceptron.this.m_numAttributes; noa++) {
				MultilayerPerceptron.this.m_inputs[noa].drawInputLines(g, x, y);
			}
			for (int noa = 0; noa < MultilayerPerceptron.this.m_numClasses; noa++) {
				MultilayerPerceptron.this.m_outputs[noa].drawInputLines(g, x, y);
				MultilayerPerceptron.this.m_outputs[noa].drawOutputLines(g, x, y);
			}
			for (NeuralConnection m_neuralNode : MultilayerPerceptron.this.m_neuralNodes) {
				m_neuralNode.drawInputLines(g, x, y);
			}
			for (int noa = 0; noa < MultilayerPerceptron.this.m_numAttributes; noa++) {
				MultilayerPerceptron.this.m_inputs[noa].drawNode(g, x, y);
			}
			for (int noa = 0; noa < MultilayerPerceptron.this.m_numClasses; noa++) {
				MultilayerPerceptron.this.m_outputs[noa].drawNode(g, x, y);
			}
			for (NeuralConnection m_neuralNode : MultilayerPerceptron.this.m_neuralNodes) {
				m_neuralNode.drawNode(g, x, y);
			}

			for (int noa = 0; noa < MultilayerPerceptron.this.m_selected.size(); noa++) {
				MultilayerPerceptron.this.m_selected.get(noa).drawHighlight(g, x, y);
			}
		}

		/**
		 * Returns the revision string.
		 *
		 * @return the revision
		 */
		@Override
		public String getRevision() {
			return RevisionUtils.extract("$Revision$");
		}
	}

	/**
	 * This provides the basic controls for working with the neuralnetwork
	 *
	 * @author Malcolm Ware (mfw4@cs.waikato.ac.nz)
	 * @version $Revision$
	 */
	class ControlPanel extends JPanel implements RevisionHandler {

		/** for serialization */
		static final long serialVersionUID = 7393543302294142271L;

		/** The start stop button. */
		public JButton m_startStop;

		/** The button to accept the network (even if it hasn't done all epochs. */
		public JButton m_acceptButton;

		/** A label to state the number of epochs processed so far. */
		public JPanel m_epochsLabel;

		/** A label to state the total number of epochs to be processed. */
		public JLabel m_totalEpochsLabel;

		/** A text field to allow the changing of the total number of epochs. */
		public JTextField m_changeEpochs;

		/** A label to state the learning rate. */
		public JLabel m_learningLabel;

		/** A label to state the momentum. */
		public JLabel m_momentumLabel;

		/** A text field to allow the changing of the learning rate. */
		public JTextField m_changeLearning;

		/** A text field to allow the changing of the momentum. */
		public JTextField m_changeMomentum;

		/**
		 * A label to state roughly the accuracy of the network.(because the accuracy is calculated per
		 * epoch, but the network is changing throughout each epoch train).
		 */
		public JPanel m_errorLabel;

		/** The constructor. */
		public ControlPanel() {
			this.setBorder(BorderFactory.createTitledBorder("Controls"));

			this.m_totalEpochsLabel = new JLabel("Num Of Epochs  ");
			this.m_epochsLabel = new JPanel() {
				/** for serialization */
				private static final long serialVersionUID = 2562773937093221399L;

				@Override
				public void paintComponent(final Graphics g) {
					super.paintComponent(g);
					g.setColor(MultilayerPerceptron.this.m_controlPanel.m_totalEpochsLabel.getForeground());
					g.drawString("Epoch  " + MultilayerPerceptron.this.m_epoch, 0, 10);
				}
			};
			this.m_epochsLabel.setFont(this.m_totalEpochsLabel.getFont());

			this.m_changeEpochs = new JTextField();
			this.m_changeEpochs.setText("" + MultilayerPerceptron.this.m_numEpochs);
			this.m_errorLabel = new JPanel() {
				/** for serialization */
				private static final long serialVersionUID = 4390239056336679189L;

				@Override
				public void paintComponent(final Graphics g) {
					super.paintComponent(g);
					g.setColor(MultilayerPerceptron.this.m_controlPanel.m_totalEpochsLabel.getForeground());
					if (MultilayerPerceptron.this.m_valSize == 0) {
						g.drawString("Error per Epoch = " + Utils.doubleToString(MultilayerPerceptron.this.m_error, 7), 0, 10);
					} else {
						g.drawString("Validation Error per Epoch = " + Utils.doubleToString(MultilayerPerceptron.this.m_error, 7), 0, 10);
					}
				}
			};
			this.m_errorLabel.setFont(this.m_epochsLabel.getFont());

			this.m_learningLabel = new JLabel("Learning Rate = ");
			this.m_momentumLabel = new JLabel("Momentum = ");
			this.m_changeLearning = new JTextField();
			this.m_changeMomentum = new JTextField();
			this.m_changeLearning.setText("" + MultilayerPerceptron.this.m_learningRate);
			this.m_changeMomentum.setText("" + MultilayerPerceptron.this.m_momentum);
			this.setLayout(new BorderLayout(15, 10));

			MultilayerPerceptron.this.m_stopIt = true;
			MultilayerPerceptron.this.m_accepted = false;
			this.m_startStop = new JButton("Start");
			this.m_startStop.setActionCommand("Start");

			this.m_acceptButton = new JButton("Accept");
			this.m_acceptButton.setActionCommand("Accept");

			JPanel buttons = new JPanel();
			buttons.setLayout(new BoxLayout(buttons, BoxLayout.Y_AXIS));
			buttons.add(this.m_startStop);
			buttons.add(this.m_acceptButton);
			this.add(buttons, BorderLayout.WEST);
			JPanel data = new JPanel();
			data.setLayout(new BoxLayout(data, BoxLayout.Y_AXIS));

			Box ab = new Box(BoxLayout.X_AXIS);
			ab.add(this.m_epochsLabel);
			data.add(ab);

			ab = new Box(BoxLayout.X_AXIS);
			Component b = Box.createGlue();
			ab.add(this.m_totalEpochsLabel);
			ab.add(this.m_changeEpochs);
			this.m_changeEpochs.setMaximumSize(new Dimension(200, 20));
			ab.add(b);
			data.add(ab);

			ab = new Box(BoxLayout.X_AXIS);
			ab.add(this.m_errorLabel);
			data.add(ab);

			this.add(data, BorderLayout.CENTER);

			data = new JPanel();
			data.setLayout(new BoxLayout(data, BoxLayout.Y_AXIS));
			ab = new Box(BoxLayout.X_AXIS);
			b = Box.createGlue();
			ab.add(this.m_learningLabel);
			ab.add(this.m_changeLearning);
			this.m_changeLearning.setMaximumSize(new Dimension(200, 20));
			ab.add(b);
			data.add(ab);

			ab = new Box(BoxLayout.X_AXIS);
			b = Box.createGlue();
			ab.add(this.m_momentumLabel);
			ab.add(this.m_changeMomentum);
			this.m_changeMomentum.setMaximumSize(new Dimension(200, 20));
			ab.add(b);
			data.add(ab);

			this.add(data, BorderLayout.EAST);

			this.m_startStop.addActionListener(new ActionListener() {
				@Override
				public void actionPerformed(final ActionEvent e) {
					if (e.getActionCommand().equals("Start")) {
						MultilayerPerceptron.this.m_stopIt = false;
						ControlPanel.this.m_startStop.setText("Stop");
						ControlPanel.this.m_startStop.setActionCommand("Stop");
						int n = Integer.valueOf(ControlPanel.this.m_changeEpochs.getText()).intValue();

						MultilayerPerceptron.this.m_numEpochs = n;
						ControlPanel.this.m_changeEpochs.setText("" + MultilayerPerceptron.this.m_numEpochs);

						double m = Double.valueOf(ControlPanel.this.m_changeLearning.getText()).doubleValue();
						MultilayerPerceptron.this.setLearningRate(m);
						ControlPanel.this.m_changeLearning.setText("" + MultilayerPerceptron.this.m_learningRate);

						m = Double.valueOf(ControlPanel.this.m_changeMomentum.getText()).doubleValue();
						MultilayerPerceptron.this.setMomentum(m);
						ControlPanel.this.m_changeMomentum.setText("" + MultilayerPerceptron.this.m_momentum);

						MultilayerPerceptron.this.blocker(false);
					} else if (e.getActionCommand().equals("Stop")) {
						MultilayerPerceptron.this.m_stopIt = true;
						ControlPanel.this.m_startStop.setText("Start");
						ControlPanel.this.m_startStop.setActionCommand("Start");
					}
				}
			});

			this.m_acceptButton.addActionListener(new ActionListener() {
				@Override
				public void actionPerformed(final ActionEvent e) {
					MultilayerPerceptron.this.m_accepted = true;
					MultilayerPerceptron.this.blocker(false);
				}
			});

			this.m_changeEpochs.addActionListener(new ActionListener() {
				@Override
				public void actionPerformed(final ActionEvent e) {
					int n = Integer.valueOf(ControlPanel.this.m_changeEpochs.getText()).intValue();
					if (n > 0) {
						MultilayerPerceptron.this.m_numEpochs = n;
						MultilayerPerceptron.this.blocker(false);
					}
				}
			});
		}

		/**
		 * Returns the revision string.
		 *
		 * @return the revision
		 */
		@Override
		public String getRevision() {
			return RevisionUtils.extract("$Revision$");
		}
	}

	/**
	 * a ZeroR model in case no model can be built from the data or the network predicts all zeros for
	 * the classes
	 */
	private Classifier m_ZeroR;

	/** Whether to use the default ZeroR model */
	private boolean m_useDefaultModel = false;

	/** The training instances. */
	private Instances m_instances;

	/** The current instance running through the network. */
	private Instance m_currentInstance;

	/** A flag to say that it's a numeric class. */
	private boolean m_numeric;

	/** The ranges for all the attributes. */
	private double[] m_attributeRanges;

	/** The base values for all the attributes. */
	private double[] m_attributeBases;

	/** The output units.(only feeds the errors, does no calcs) */
	private NeuralEnd[] m_outputs;

	/** The input units.(only feeds the inputs does no calcs) */
	private NeuralEnd[] m_inputs;

	/** All the nodes that actually comprise the logical neural net. */
	private NeuralConnection[] m_neuralNodes;

	/** The number of classes. */
	private int m_numClasses = 0;

	/** The number of attributes. */
	private int m_numAttributes = 0; // note the number doesn't include the class.

	/** The panel the nodes are displayed on. */
	private NodePanel m_nodePanel;

	/** The control panel. */
	private ControlPanel m_controlPanel;

	/** The next id number available for default naming. */
	private int m_nextId;

	/** A Vector list of the units currently selected. */
	private ArrayList<NeuralConnection> m_selected;

	/** The number of epochs to train through. */
	private int m_numEpochs;

	/** a flag to state if the network should be running, or stopped. */
	private boolean m_stopIt;

	/** a flag to state that the network has in fact stopped. */
	private boolean m_stopped;

	/** a flag to state that the network should be accepted the way it is. */
	private boolean m_accepted;
	/** The window for the network. */
	private JFrame m_win;

	/**
	 * A flag to tell the build classifier to automatically build a neural net.
	 */
	private boolean m_autoBuild;

	/**
	 * A flag to state that the gui for the network should be brought up. To allow interaction while
	 * training.
	 */
	private boolean m_gui;

	/** An int to say how big the validation set should be. */
	private int m_valSize;

	/** The number to to use to quit on validation testing. */
	private int m_driftThreshold;

	/** The number used to seed the random number generator. */
	private int m_randomSeed;

	/** The actual random number generator. */
	private Random m_random;

	/** A flag to state that a nominal to binary filter should be used. */
	private boolean m_useNomToBin;

	/** The actual filter. */
	private NominalToBinary m_nominalToBinaryFilter;

	/** The string that defines the hidden layers */
	private String m_hiddenLayers;

	/** This flag states that the user wants the input values normalized. */
	private boolean m_normalizeAttributes;

	/** This flag states that the user wants the learning rate to decay. */
	private boolean m_decay;

	/** This is the learning rate for the network. */
	private double m_learningRate;

	/** This is the momentum for the network. */
	private double m_momentum;

	/** Shows the number of the epoch that the network just finished. */
	private int m_epoch;

	/** Shows the error of the epoch that the network just finished. */
	private double m_error;

	/**
	 * This flag states that the user wants the network to restart if it is found to be generating
	 * infinity or NaN for the error value. This would restart the network with the current options
	 * except that the learning rate would be smaller than before, (perhaps half of its current value).
	 * This option will not be available if the gui is chosen (if the gui is open the user can fix the
	 * network themselves, it is an architectural minefield for the network to be reset with the gui
	 * open).
	 */
	private boolean m_reset;

	/**
	 * This flag states that the user wants the class to be normalized while processing in the network
	 * is done. (the final answer will be in the original range regardless). This option will only be
	 * used when the class is numeric.
	 */
	private boolean m_normalizeClass;

	/**
	 * this is a sigmoid unit.
	 */
	private final SigmoidUnit m_sigmoidUnit;

	/**
	 * This is a linear unit.
	 */
	private final LinearUnit m_linearUnit;

	/**
	 * The constructor.
	 */
	public MultilayerPerceptron() {
		this.m_instances = null;
		this.m_currentInstance = null;
		this.m_controlPanel = null;
		this.m_nodePanel = null;
		this.m_epoch = 0;
		this.m_error = 0;

		this.m_outputs = new NeuralEnd[0];
		this.m_inputs = new NeuralEnd[0];
		this.m_numAttributes = 0;
		this.m_numClasses = 0;
		this.m_neuralNodes = new NeuralConnection[0];
		this.m_selected = new ArrayList<>(4);
		this.m_nextId = 0;
		this.m_stopIt = true;
		this.m_stopped = true;
		this.m_accepted = false;
		this.m_numeric = false;
		this.m_random = null;
		this.m_nominalToBinaryFilter = new NominalToBinary();
		this.m_sigmoidUnit = new SigmoidUnit();
		this.m_linearUnit = new LinearUnit();
		// setting all the options to their defaults. To completely change these
		// defaults they will also need to be changed down the bottom in the
		// setoptions function (the text info in the accompanying functions should
		// also be changed to reflect the new defaults
		this.m_normalizeClass = true;
		this.m_normalizeAttributes = true;
		this.m_autoBuild = true;
		this.m_gui = false;
		this.m_useNomToBin = true;
		this.m_driftThreshold = 20;
		this.m_numEpochs = 500;
		this.m_valSize = 0;
		this.m_randomSeed = 0;
		this.m_hiddenLayers = "a";
		this.m_learningRate = .3;
		this.m_momentum = .2;
		this.m_reset = true;
		this.m_decay = false;
	}

	/**
	 * @param d
	 *          True if the learning rate should decay.
	 */
	public void setDecay(final boolean d) {
		this.m_decay = d;
	}

	/**
	 * @return the flag for having the learning rate decay.
	 */
	public boolean getDecay() {
		return this.m_decay;
	}

	/**
	 * This sets the network up to be able to reset itself with the current settings and the learning
	 * rate at half of what it is currently. This will only happen if the network creates NaN or
	 * infinite errors. Also this will continue to happen until the network is trained properly. The
	 * learning rate will also get set back to it's original value at the end of this. This can only be
	 * set to true if the GUI is not brought up.
	 *
	 * @param r
	 *          True if the network should restart with it's current options and set the learning rate
	 *          to half what it currently is.
	 */
	public void setReset(boolean r) {
		if (this.m_gui) {
			r = false;
		}
		this.m_reset = r;

	}

	/**
	 * @return The flag for reseting the network.
	 */
	public boolean getReset() {
		return this.m_reset;
	}

	/**
	 * @param c
	 *          True if the class should be normalized (the class will only ever be normalized if it is
	 *          numeric). (Normalization puts the range between -1 - 1).
	 */
	public void setNormalizeNumericClass(final boolean c) {
		this.m_normalizeClass = c;
	}

	/**
	 * @return The flag for normalizing a numeric class.
	 */
	public boolean getNormalizeNumericClass() {
		return this.m_normalizeClass;
	}

	/**
	 * @param a
	 *          True if the attributes should be normalized (even nominal attributes will get normalized
	 *          here) (range goes between -1 - 1).
	 */
	public void setNormalizeAttributes(final boolean a) {
		this.m_normalizeAttributes = a;
	}

	/**
	 * @return The flag for normalizing attributes.
	 */
	public boolean getNormalizeAttributes() {
		return this.m_normalizeAttributes;
	}

	/**
	 * @param f
	 *          True if a nominalToBinary filter should be used on the data.
	 */
	public void setNominalToBinaryFilter(final boolean f) {
		this.m_useNomToBin = f;
	}

	/**
	 * @return The flag for nominal to binary filter use.
	 */
	public boolean getNominalToBinaryFilter() {
		return this.m_useNomToBin;
	}

	/**
	 * This seeds the random number generator, that is used when a random number is needed for the
	 * network.
	 *
	 * @param l
	 *          The seed.
	 */
	@Override
	public void setSeed(final int l) {
		if (l >= 0) {
			this.m_randomSeed = l;
		}
	}

	/**
	 * @return The seed for the random number generator.
	 */
	@Override
	public int getSeed() {
		return this.m_randomSeed;
	}

	/**
	 * This sets the threshold to use for when validation testing is being done. It works by ending
	 * testing once the error on the validation set has consecutively increased a certain number of
	 * times.
	 *
	 * @param t
	 *          The threshold to use for this.
	 */
	public void setValidationThreshold(final int t) {
		if (t > 0) {
			this.m_driftThreshold = t;
		}
	}

	/**
	 * @return The threshold used for validation testing.
	 */
	public int getValidationThreshold() {
		return this.m_driftThreshold;
	}

	/**
	 * The learning rate can be set using this command. NOTE That this is a static variable so it affect
	 * all networks that are running. Must be greater than 0 and no more than 1.
	 *
	 * @param l
	 *          The New learning rate.
	 */
	public void setLearningRate(final double l) {
		if (l > 0 && l <= 1) {
			this.m_learningRate = l;

			if (this.m_controlPanel != null) {
				this.m_controlPanel.m_changeLearning.setText("" + l);
			}
		}
	}

	/**
	 * @return The learning rate for the nodes.
	 */
	public double getLearningRate() {
		return this.m_learningRate;
	}

	/**
	 * The momentum can be set using this command. THE same conditions apply to this as to the learning
	 * rate.
	 *
	 * @param m
	 *          The new Momentum.
	 */
	public void setMomentum(final double m) {
		if (m >= 0 && m <= 1) {
			this.m_momentum = m;

			if (this.m_controlPanel != null) {
				this.m_controlPanel.m_changeMomentum.setText("" + m);
			}
		}
	}

	/**
	 * @return The momentum for the nodes.
	 */
	public double getMomentum() {
		return this.m_momentum;
	}

	/**
	 * This will set whether the network is automatically built or if it is left up to the user. (there
	 * is nothing to stop a user from altering an autobuilt network however).
	 *
	 * @param a
	 *          True if the network should be auto built.
	 */
	public void setAutoBuild(boolean a) {
		if (!this.m_gui) {
			a = true;
		}
		this.m_autoBuild = a;
	}

	/**
	 * @return The auto build state.
	 */
	public boolean getAutoBuild() {
		return this.m_autoBuild;
	}

	/**
	 * This will set what the hidden layers are made up of when auto build is enabled. Note to have no
	 * hidden units, just put a single 0, Any more 0's will indicate that the string is badly formed and
	 * make it unaccepted. Negative numbers, and floats will do the same. There are also some wildcards.
	 * These are 'a' = (number of attributes + number of classes) / 2, 'i' = number of attributes, 'o' =
	 * number of classes, and 't' = number of attributes + number of classes.
	 *
	 * @param h
	 *          A string with a comma seperated list of numbers. Each number is the number of nodes to
	 *          be on a hidden layer.
	 */
	public void setHiddenLayers(final String h) {
		String tmp = "";
		StringTokenizer tok = new StringTokenizer(h, ",");
		if (tok.countTokens() == 0) {
			return;
		}
		double dval;
		int val;
		String c;
		boolean first = true;
		while (tok.hasMoreTokens()) {
			c = tok.nextToken().trim();

			if (c.equals("a") || c.equals("i") || c.equals("o") || c.equals("t")) {
				tmp += c;
			} else {
				dval = Double.valueOf(c).doubleValue();
				val = (int) dval;

				if ((val == dval && (val != 0 || (tok.countTokens() == 0 && first)) && val >= 0)) {
					tmp += val;
				} else {
					return;
				}
			}

			first = false;
			if (tok.hasMoreTokens()) {
				tmp += ", ";
			}
		}
		this.m_hiddenLayers = tmp;
	}

	/**
	 * @return A string representing the hidden layers, each number is the number of nodes on a hidden
	 *         layer.
	 */
	public String getHiddenLayers() {
		return this.m_hiddenLayers;
	}

	/**
	 * This will set whether A GUI is brought up to allow interaction by the user with the neural
	 * network during training.
	 *
	 * @param a
	 *          True if gui should be created.
	 */
	public void setGUI(final boolean a) {
		this.m_gui = a;
		if (!a) {
			this.setAutoBuild(true);

		} else {
			this.setReset(false);
		}
	}

	/**
	 * @return The true if should show gui.
	 */
	public boolean getGUI() {
		return this.m_gui;
	}

	/**
	 * This will set the size of the validation set.
	 *
	 * @param a
	 *          The size of the validation set, as a percentage of the whole.
	 */
	public void setValidationSetSize(final int a) {
		if (a < 0 || a > 99) {
			return;
		}
		this.m_valSize = a;
	}

	/**
	 * @return The percentage size of the validation set.
	 */
	public int getValidationSetSize() {
		return this.m_valSize;
	}

	/**
	 * Set the number of training epochs to perform. Must be greater than 0.
	 *
	 * @param n
	 *          The number of epochs to train through.
	 */
	public void setTrainingTime(final int n) {
		if (n > 0) {
			this.m_numEpochs = n;
		}
	}

	/**
	 * @return The number of epochs to train through.
	 */
	public int getTrainingTime() {
		return this.m_numEpochs;
	}

	/**
	 * Call this function to place a node into the network list.
	 *
	 * @param n
	 *          The node to place in the list.
	 */
	private void addNode(final NeuralConnection n) {

		NeuralConnection[] temp1 = new NeuralConnection[this.m_neuralNodes.length + 1];
		for (int noa = 0; noa < this.m_neuralNodes.length; noa++) {
			temp1[noa] = this.m_neuralNodes[noa];
		}

		temp1[temp1.length - 1] = n;
		this.m_neuralNodes = temp1;
	}

	/**
	 * Call this function to remove the passed node from the list. This will only remove the node if it
	 * is in the neuralnodes list.
	 *
	 * @param n
	 *          The neuralConnection to remove.
	 * @return True if removed false if not (because it wasn't there).
	 */
	private boolean removeNode(final NeuralConnection n) {
		NeuralConnection[] temp1 = new NeuralConnection[this.m_neuralNodes.length - 1];
		int skip = 0;
		for (int noa = 0; noa < this.m_neuralNodes.length; noa++) {
			if (n == this.m_neuralNodes[noa]) {
				skip++;
			} else if (!((noa - skip) >= temp1.length)) {
				temp1[noa - skip] = this.m_neuralNodes[noa];
			} else {
				return false;
			}
		}
		this.m_neuralNodes = temp1;
		return true;
	}

	/**
	 * This function sets what the m_numeric flag to represent the passed class it also performs the
	 * normalization of the attributes if applicable and sets up the info to normalize the class. (note
	 * that regardless of the options it will fill an array with the range and base, set to normalize
	 * all attributes and the class to be between -1 and 1)
	 *
	 * @param inst
	 *          the instances.
	 * @return The modified instances. This needs to be done. If the attributes are normalized then deep
	 *         copies will be made of all the instances which will need to be passed back out.
	 */
	private Instances setClassType(final Instances inst) throws Exception {
		if (inst != null) {
			// x bounds
			this.m_attributeRanges = new double[inst.numAttributes()];
			this.m_attributeBases = new double[inst.numAttributes()];
			for (int noa = 0; noa < inst.numAttributes(); noa++) {
				double min = Double.POSITIVE_INFINITY;
				double max = Double.NEGATIVE_INFINITY;
				for (int i = 0; i < inst.numInstances(); i++) {
					if (!inst.instance(i).isMissing(noa)) {
						double value = inst.instance(i).value(noa);
						if (value < min) {
							min = value;
						}
						if (value > max) {
							max = value;
						}
					}
				}
				this.m_attributeRanges[noa] = (max - min) / 2;
				this.m_attributeBases[noa] = (max + min) / 2;
			}

			if (this.m_normalizeAttributes) {
				for (int i = 0; i < inst.numInstances(); i++) {
					Instance currentInstance = inst.instance(i);
					double[] instance = new double[inst.numAttributes()];
					for (int noa = 0; noa < inst.numAttributes(); noa++) {
						if (noa != inst.classIndex()) {
							if (this.m_attributeRanges[noa] != 0) {
								instance[noa] = (currentInstance.value(noa) - this.m_attributeBases[noa]) / this.m_attributeRanges[noa];
							} else {
								instance[noa] = currentInstance.value(noa) - this.m_attributeBases[noa];
							}
						} else {
							instance[noa] = currentInstance.value(noa);
						}
					}
					inst.set(i, new DenseInstance(currentInstance.weight(), instance));
				}
			}

			if (inst.classAttribute().isNumeric()) {
				this.m_numeric = true;
			} else {
				this.m_numeric = false;
			}
		}
		return inst;
	}

	/**
	 * A function used to stop the code that called buildclassifier from continuing on before the user
	 * has finished the decision tree.
	 *
	 * @param tf
	 *          True to stop the thread, False to release the thread that is waiting there (if one).
	 */
	public synchronized void blocker(final boolean tf) {
		if (tf) {
			try {
				this.wait();
			} catch (InterruptedException e) {
			}
		} else {
			this.notifyAll();
		}
	}

	/**
	 * Call this function to update the control panel for the gui.
	 */
	private void updateDisplay() {

		if (this.m_gui) {
			this.m_controlPanel.m_errorLabel.repaint();
			this.m_controlPanel.m_epochsLabel.repaint();
		}
	}

	/**
	 * this will reset all the nodes in the network.
	 */
	private void resetNetwork() {
		for (int noc = 0; noc < this.m_numClasses; noc++) {
			this.m_outputs[noc].reset();
		}
	}

	/**
	 * This will cause the output values of all the nodes to be calculated. Note that the
	 * m_currentInstance is used to calculate these values.
	 */
	private void calculateOutputs() {
		for (int noc = 0; noc < this.m_numClasses; noc++) {
			// get the values.
			this.m_outputs[noc].outputValue(true);
		}
	}

	/**
	 * This will cause the error values to be calculated for all nodes. Note that the m_currentInstance
	 * is used to calculate these values. Also the output values should have been calculated first.
	 *
	 * @return The squared error.
	 */
	private double calculateErrors() throws Exception {
		double ret = 0, temp = 0;
		for (int noc = 0; noc < this.m_numAttributes; noc++) {
			// get the errors.
			this.m_inputs[noc].errorValue(true);

		}
		for (int noc = 0; noc < this.m_numClasses; noc++) {
			temp = this.m_outputs[noc].errorValue(false);
			ret += temp * temp;
		}
		return ret;

	}

	/**
	 * This will cause the weight values to be updated based on the learning rate, momentum and the
	 * errors that have been calculated for each node.
	 *
	 * @param l
	 *          The learning rate to update with.
	 * @param m
	 *          The momentum to update with.
	 */
	private void updateNetworkWeights(final double l, final double m) {
		for (int noc = 0; noc < this.m_numClasses; noc++) {
			// update weights
			this.m_outputs[noc].updateWeights(l, m);
		}

	}

	/**
	 * This creates the required input units.
	 */
	private void setupInputs() throws Exception {
		this.m_inputs = new NeuralEnd[this.m_numAttributes];
		int now = 0;
		for (int noa = 0; noa < this.m_numAttributes + 1; noa++) {
			if (this.m_instances.classIndex() != noa) {
				this.m_inputs[noa - now] = new NeuralEnd(this.m_instances.attribute(noa).name());

				this.m_inputs[noa - now].setX(.1);
				this.m_inputs[noa - now].setY((noa - now + 1.0) / (this.m_numAttributes + 1));
				this.m_inputs[noa - now].setLink(true, noa);
			} else {
				now = 1;
			}
		}

	}

	/**
	 * This creates the required output units.
	 */
	private void setupOutputs() throws Exception {

		this.m_outputs = new NeuralEnd[this.m_numClasses];
		for (int noa = 0; noa < this.m_numClasses; noa++) {
			if (this.m_numeric) {
				this.m_outputs[noa] = new NeuralEnd(this.m_instances.classAttribute().name());
			} else {
				this.m_outputs[noa] = new NeuralEnd(this.m_instances.classAttribute().value(noa));
			}

			this.m_outputs[noa].setX(.9);
			this.m_outputs[noa].setY((noa + 1.0) / (this.m_numClasses + 1));
			this.m_outputs[noa].setLink(false, noa);
			NeuralNode temp = new NeuralNode(String.valueOf(this.m_nextId), this.m_random, this.m_sigmoidUnit);
			this.m_nextId++;
			temp.setX(.75);
			temp.setY((noa + 1.0) / (this.m_numClasses + 1));
			this.addNode(temp);
			NeuralConnection.connect(temp, this.m_outputs[noa]);
		}

	}

	/**
	 * Call this function to automatically generate the hidden units
	 *
	 * @throws InterruptedException
	 */
	private void setupHiddenLayer() throws InterruptedException {
		StringTokenizer tok = new StringTokenizer(this.m_hiddenLayers, ",");
		int val = 0; // num of nodes in a layer
		int prev = 0; // used to remember the previous layer
		int num = tok.countTokens(); // number of layers
		String c;
		for (int noa = 0; noa < num; noa++) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			// note that I am using the Double to get the value rather than the
			// Integer class, because for some reason the Double implementation can
			// handle leading white space and the integer version can't!?!
			c = tok.nextToken().trim();
			if (c.equals("a")) {
				val = (this.m_numAttributes + this.m_numClasses) / 2;
			} else if (c.equals("i")) {
				val = this.m_numAttributes;
			} else if (c.equals("o")) {
				val = this.m_numClasses;
			} else if (c.equals("t")) {
				val = this.m_numAttributes + this.m_numClasses;
			} else {
				val = Double.valueOf(c).intValue();
			}
			for (int nob = 0; nob < val; nob++) {
				// XXX kill weka execution
				if (Thread.interrupted()) {
					throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
				}
				NeuralNode temp = new NeuralNode(String.valueOf(this.m_nextId), this.m_random, this.m_sigmoidUnit);
				this.m_nextId++;
				temp.setX(.5 / (num) * noa + .25);
				temp.setY((nob + 1.0) / (val + 1));
				this.addNode(temp);
				if (noa > 0) {
					// then do connections
					for (int noc = this.m_neuralNodes.length - nob - 1 - prev; noc < this.m_neuralNodes.length - nob - 1; noc++) {
						NeuralConnection.connect(this.m_neuralNodes[noc], temp);
					}
				}
			}
			prev = val;
		}
		tok = new StringTokenizer(this.m_hiddenLayers, ",");
		c = tok.nextToken();
		if (c.equals("a")) {
			val = (this.m_numAttributes + this.m_numClasses) / 2;
		} else if (c.equals("i")) {
			val = this.m_numAttributes;
		} else if (c.equals("o")) {
			val = this.m_numClasses;
		} else if (c.equals("t")) {
			val = this.m_numAttributes + this.m_numClasses;
		} else {
			val = Double.valueOf(c).intValue();
		}

		if (val == 0) {
			for (int noa = 0; noa < this.m_numAttributes; noa++) {
				for (int nob = 0; nob < this.m_numClasses; nob++) {
					// XXX kill weka execution
					if (Thread.interrupted()) {
						throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
					}
					NeuralConnection.connect(this.m_inputs[noa], this.m_neuralNodes[nob]);
				}
			}
		} else {
			for (int noa = 0; noa < this.m_numAttributes; noa++) {
				for (int nob = this.m_numClasses; nob < this.m_numClasses + val; nob++) {
					// XXX kill weka execution
					if (Thread.interrupted()) {
						throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
					}
					NeuralConnection.connect(this.m_inputs[noa], this.m_neuralNodes[nob]);
				}
			}
			for (int noa = this.m_neuralNodes.length - prev; noa < this.m_neuralNodes.length; noa++) {
				for (int nob = 0; nob < this.m_numClasses; nob++) {
					// XXX kill weka execution
					if (Thread.interrupted()) {
						throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
					}
					NeuralConnection.connect(this.m_neuralNodes[noa], this.m_neuralNodes[nob]);
				}
			}
		}

	}

	/**
	 * This will go through all the nodes and check if they are connected to a pure output unit. If so
	 * they will be set to be linear units. If not they will be set to be sigmoid units.
	 */
	private void setEndsToLinear() {
		for (NeuralConnection m_neuralNode : this.m_neuralNodes) {
			if ((m_neuralNode.getType() & NeuralConnection.OUTPUT) == NeuralConnection.OUTPUT) {
				((NeuralNode) m_neuralNode).setMethod(this.m_linearUnit);
			} else {
				((NeuralNode) m_neuralNode).setMethod(this.m_sigmoidUnit);
			}
		}
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.DATE_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		return result;
	}

	/** The instances in the validation set (if any) */
	protected transient Instances valSet = null;

	/** The number of instances in the validation set (if any) */
	protected transient int numInVal = 0;

	/** Total weight of the instances in the training set */
	protected transient double totalWeight = 0;

	/** Total weight of the instances in the validation set (if any) */
	protected transient double totalValWeight = 0;

	/** Drift off counter */
	protected transient double driftOff = 0;

	/** To keep track of error */
	protected transient double lastRight = Double.POSITIVE_INFINITY;
	protected transient double bestError = Double.POSITIVE_INFINITY;

	/** Data in original format (in case learning rate gets reset */
	protected transient Instances originalFormatData = null;

	/**
	 * Initializes an iterative classifier.
	 *
	 * @param data
	 *          the instances to be used in induction
	 * @exception Exception
	 *              if the model cannot be initialized
	 */
	@Override
	public void initializeClassifier(Instances data) throws Exception {

		// can classifier handle the data?
		this.getCapabilities().testWithFail(data);

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();
		this.originalFormatData = data;

		this.m_ZeroR = new weka.classifiers.rules.ZeroR();
		this.m_ZeroR.buildClassifier(data);
		// only class? -> use ZeroR model
		if (data.numAttributes() == 1) {
			System.err.println("Cannot build model (only class attribute present in data!), " + "using ZeroR model instead!");
			this.m_useDefaultModel = true;
			return;
		} else {
			this.m_useDefaultModel = false;
		}

		this.m_epoch = 0;
		this.m_error = 0;
		this.m_instances = null;
		this.m_currentInstance = null;
		this.m_controlPanel = null;
		this.m_nodePanel = null;

		this.m_outputs = new NeuralEnd[0];
		this.m_inputs = new NeuralEnd[0];
		this.m_numAttributes = 0;
		this.m_numClasses = 0;
		this.m_neuralNodes = new NeuralConnection[0];

		this.m_selected = new ArrayList<>(4);
		this.m_nextId = 0;
		this.m_stopIt = true;
		this.m_stopped = true;
		this.m_accepted = false;
		this.m_instances = new Instances(data);
		this.m_random = new Random(this.m_randomSeed);

		if (Thread.interrupted()) {
			throw new InterruptedException("Killed WEKA!");
		}
		this.m_instances.randomize(this.m_random);

		if (this.m_useNomToBin) {
			this.m_nominalToBinaryFilter = new NominalToBinary();
			this.m_nominalToBinaryFilter.setInputFormat(this.m_instances);
			this.m_instances = Filter.useFilter(this.m_instances, this.m_nominalToBinaryFilter);
		}
		this.m_numAttributes = this.m_instances.numAttributes() - 1;
		this.m_numClasses = this.m_instances.numClasses();

		this.setClassType(this.m_instances);

		// this sets up the validation set.
		// numinval is needed later
		this.numInVal = (int) (this.m_valSize / 100.0 * this.m_instances.numInstances());
		if (this.m_valSize > 0) {
			if (this.numInVal == 0) {
				this.numInVal = 1;
			}
			this.valSet = new Instances(this.m_instances, 0, this.numInVal);
		}
		// /////////

		this.setupInputs();

		this.setupOutputs();
		if (this.m_autoBuild) {
			this.setupHiddenLayer();
		}

		// ///////////////////////////
		// this sets up the gui for usage
		if (this.m_gui) {
			this.m_win = Utils.getWekaJFrame("Neural Network", null);

			this.m_win.addWindowListener(new WindowAdapter() {
				@Override
				public void windowClosing(final WindowEvent e) {
					boolean k = MultilayerPerceptron.this.m_stopIt;
					MultilayerPerceptron.this.m_stopIt = true;
					int well = JOptionPane.showConfirmDialog(MultilayerPerceptron.this.m_win, "Are You Sure...\n" + "Click Yes To Accept" + " The Neural Network" + "\n Click No To Return", "Accept Neural Network",
							JOptionPane.YES_NO_OPTION);

					if (well == 0) {
						MultilayerPerceptron.this.m_win.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
						MultilayerPerceptron.this.m_accepted = true;
						MultilayerPerceptron.this.blocker(false);
					} else {
						MultilayerPerceptron.this.m_win.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
					}
					MultilayerPerceptron.this.m_stopIt = k;
				}
			});

			this.m_win.getContentPane().setLayout(new BorderLayout());
			this.m_nodePanel = new NodePanel();
			// without the following two lines, the
			// NodePanel.paintComponents(Graphics)
			// method will go berserk if the network doesn't fit completely: it will
			// get called on a constant basis, using 100% of the CPU
			// see the following forum thread:
			// http://forum.java.sun.com/thread.jspa?threadID=580929&messageID=2945011
			this.m_nodePanel.setPreferredSize(new Dimension(640, 480));
			this.m_nodePanel.revalidate();

			JScrollPane sp = new JScrollPane(this.m_nodePanel, JScrollPane.VERTICAL_SCROLLBAR_ALWAYS, JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
			this.m_controlPanel = new ControlPanel();

			this.m_win.getContentPane().add(sp, BorderLayout.CENTER);
			this.m_win.getContentPane().add(this.m_controlPanel, BorderLayout.SOUTH);
			this.m_win.setSize(640, 480);
			this.m_win.setVisible(true);
		}

		// This sets up the initial state of the gui
		if (this.m_gui) {
			this.blocker(true);
			this.m_controlPanel.m_changeEpochs.setEnabled(false);
			this.m_controlPanel.m_changeLearning.setEnabled(false);
			this.m_controlPanel.m_changeMomentum.setEnabled(false);
		}

		// For silly situations in which the network gets accepted before training
		// commenses
		if (this.m_numeric) {
			this.setEndsToLinear();
		}
		if (this.m_accepted) {
			return;
		}

		// connections done.

		this.totalWeight = 0;
		this.totalValWeight = 0;
		this.driftOff = 0;
		this.lastRight = Double.POSITIVE_INFINITY;
		this.bestError = Double.POSITIVE_INFINITY;

		// ensure that at least 1 instance is trained through.
		if (this.numInVal == this.m_instances.numInstances()) {
			this.numInVal--;
		}
		if (this.numInVal < 0) {
			this.numInVal = 0;
		}
		for (int noa = this.numInVal; noa < this.m_instances.numInstances(); noa++) {
			if (!this.m_instances.instance(noa).classIsMissing()) {
				this.totalWeight += this.m_instances.instance(noa).weight();
			}
		}
		if (this.m_valSize != 0) {
			for (int noa = 0; noa < this.valSet.numInstances(); noa++) {
				if (!this.valSet.instance(noa).classIsMissing()) {
					this.totalValWeight += this.valSet.instance(noa).weight();
				}
			}
		}
		this.m_stopped = false;
	}

	/**
	 * Performs one iteration.
	 *
	 * @return false if no further iterations could be performed, true otherwise
	 * @exception Exception
	 *              if this iteration fails for unexpected reasons
	 */
	@Override
	public boolean next() throws Exception {

		if (this.m_accepted || this.m_useDefaultModel) { // Has user accepted the network already or do we need to use default model?
			return false;
		}
		this.m_epoch++;
		double right = 0;
		for (int nob = this.numInVal; nob < this.m_instances.numInstances(); nob++) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			this.m_currentInstance = this.m_instances.instance(nob);

			if (!this.m_currentInstance.classIsMissing()) {

				// this is where the network updating (and training occurs, for the
				// training set
				this.resetNetwork();
				this.calculateOutputs();
				double tempRate = this.m_learningRate * this.m_currentInstance.weight();
				if (this.m_decay) {
					tempRate /= this.m_epoch;
				}

				right += (this.calculateErrors() / this.m_instances.numClasses()) * this.m_currentInstance.weight();
				this.updateNetworkWeights(tempRate, this.m_momentum);
			}
		}
		right /= this.totalWeight;
		if (Double.isInfinite(right) || Double.isNaN(right)) {
			if ((!this.m_reset) || (this.originalFormatData == null)) {
				this.m_instances = null;
				throw new Exception("Network cannot train. Try restarting with a smaller learning rate.");
			} else {
				// reset the network if possible
				if (this.m_learningRate <= Utils.SMALL) {
					throw new IllegalStateException("Learning rate got too small (" + this.m_learningRate + " <= " + Utils.SMALL + ")!");
				}
				double origRate = this.m_learningRate; // only used for when reset
				this.m_learningRate /= 2;
				this.buildClassifier(this.originalFormatData);
				this.m_learningRate = origRate;
				return false;
			}
		}

		// //////////////////////do validation testing if applicable
		if (this.m_valSize != 0) {
			right = 0;
			if (this.valSet == null) {
				throw new IllegalArgumentException("Trying to use validation set but validation set is null.");
			}
			for (int nob = 0; nob < this.valSet.numInstances(); nob++) {
				// XXX kill weka execution
				if (Thread.interrupted()) {
					throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
				}
				this.m_currentInstance = this.valSet.instance(nob);
				if (!this.m_currentInstance.classIsMissing()) {
					// this is where the network updating occurs, for the validation set
					this.resetNetwork();
					this.calculateOutputs();
					right += (this.calculateErrors() / this.valSet.numClasses()) * this.m_currentInstance.weight();
					// note 'right' could be calculated here just using
					// the calculate output values. This would be faster.
					// be less modular
				}
			}

			if (right < this.lastRight) {
				if (right < this.bestError) {
					this.bestError = right;
					// save the network weights at this point
					for (int noc = 0; noc < this.m_numClasses; noc++) {
						this.m_outputs[noc].saveWeights();
					}
					this.driftOff = 0;
				}
			} else {
				this.driftOff++;
			}
			this.lastRight = right;
			if (this.driftOff > this.m_driftThreshold || this.m_epoch + 1 >= this.m_numEpochs) {
				for (int noc = 0; noc < this.m_numClasses; noc++) {
					this.m_outputs[noc].restoreWeights();
				}
				this.m_accepted = true;
			}
			right /= this.totalValWeight;
		}
		this.m_error = right;
		// shows what the neuralnet is upto if a gui exists.
		this.updateDisplay();
		// This junction controls what state the gui is in at the end of each
		// epoch, Such as if it is paused, if it is resumable etc...
		if (this.m_gui) {
			while ((this.m_stopIt || (this.m_epoch >= this.m_numEpochs && this.m_valSize == 0)) && !this.m_accepted) {
				this.m_stopIt = true;
				this.m_stopped = true;
				if (this.m_epoch >= this.m_numEpochs && this.m_valSize == 0) {

					this.m_controlPanel.m_startStop.setEnabled(false);
				} else {
					this.m_controlPanel.m_startStop.setEnabled(true);
				}
				this.m_controlPanel.m_startStop.setText("Start");
				this.m_controlPanel.m_startStop.setActionCommand("Start");
				this.m_controlPanel.m_changeEpochs.setEnabled(true);
				this.m_controlPanel.m_changeLearning.setEnabled(true);
				this.m_controlPanel.m_changeMomentum.setEnabled(true);

				this.blocker(true);
				if (this.m_numeric) {
					this.setEndsToLinear();
				}
			}
			this.m_controlPanel.m_changeEpochs.setEnabled(false);
			this.m_controlPanel.m_changeLearning.setEnabled(false);
			this.m_controlPanel.m_changeMomentum.setEnabled(false);

			this.m_stopped = false;
			// if the network has been accepted stop the training loop
			if (this.m_accepted) {
				return false;
			}
		}
		if (this.m_accepted) {
			return false;
		}
		if (this.m_epoch < this.m_numEpochs) {
			return true; // We can keep iterating
		} else {
			return false;
		}
	}

	/**
	 * Signal end of iterating, useful for any house-keeping/cleanup
	 *
	 * @exception Exception
	 *              if cleanup fails
	 */
	@Override
	public void done() throws Exception {

		if (this.m_gui) {
			this.m_win.dispose();
			this.m_controlPanel = null;
			this.m_nodePanel = null;
		}
		if (!this.m_useDefaultModel) {
			this.m_instances = new Instances(this.m_instances, 0);
		}
		this.m_currentInstance = null;
		this.originalFormatData = null;
	}

	/**
	 * Call this function to build and train a neural network for the training data provided.
	 *
	 * @param i
	 *          The training data.
	 * @throws Exception
	 *           if can't build classification properly.
	 */
	@Override
	public void buildClassifier(final Instances i) throws Exception {

		// Initialize classifier
		this.initializeClassifier(i);

		// For the given number of iterations
		while (this.next()) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
		}

		// Clean up
		this.done();
	}

	/**
	 * Call this function to predict the class of an instance once a classification model has been built
	 * with the buildClassifier call.
	 *
	 * @param i
	 *          The instance to classify.
	 * @return A double array filled with the probabilities of each class type.
	 * @throws Exception
	 *           if can't classify instance.
	 */
	@Override
	public double[] distributionForInstance(final Instance i) throws Exception {

		// default model?
		if (this.m_useDefaultModel) {
			return this.m_ZeroR.distributionForInstance(i);
		}

		if (this.m_useNomToBin) {
			this.m_nominalToBinaryFilter.input(i);
			this.m_currentInstance = this.m_nominalToBinaryFilter.output();
		} else {
			this.m_currentInstance = i;
		}

		// Make a copy of the instance so that it isn't modified
		this.m_currentInstance = (Instance) this.m_currentInstance.copy();

		if (this.m_normalizeAttributes) {
			double[] instance = new double[this.m_currentInstance.numAttributes()];
			for (int noa = 0; noa < this.m_instances.numAttributes(); noa++) {
				// XXX kill weka execution
				if (Thread.interrupted()) {
					throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
				}
				if (noa != this.m_instances.classIndex()) {
					if (this.m_attributeRanges[noa] != 0) {
						instance[noa] = (this.m_currentInstance.value(noa) - this.m_attributeBases[noa]) / this.m_attributeRanges[noa];
					} else {
						instance[noa] = this.m_currentInstance.value(noa) - this.m_attributeBases[noa];
					}
				} else {
					instance[noa] = this.m_currentInstance.value(noa);
				}
			}
			this.m_currentInstance = new DenseInstance(this.m_currentInstance.weight(), instance);
			this.m_currentInstance.setDataset(this.m_instances);
		}
		this.resetNetwork();

		// since all the output values are needed.
		// They are calculated manually here and the values collected.
		double[] theArray = new double[this.m_numClasses];
		for (int noa = 0; noa < this.m_numClasses; noa++) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			theArray[noa] = this.m_outputs[noa].outputValue(true);
		}
		if (this.m_instances.classAttribute().isNumeric()) {
			return theArray;
		}

		// now normalize the array
		double count = 0;
		for (int noa = 0; noa < this.m_numClasses; noa++) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			count += theArray[noa];
		}
		if (count <= 0) {
			return this.m_ZeroR.distributionForInstance(i);
		}
		for (int noa = 0; noa < this.m_numClasses; noa++) {
			theArray[noa] /= count;
		}
		return theArray;
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<>(14);

		newVector.addElement(new Option("\tLearning Rate for the backpropagation algorithm.\n" + "\t(Value should be between 0 - 1, Default = 0.3).", "L", 1, "-L <learning rate>"));
		newVector.addElement(new Option("\tMomentum Rate for the backpropagation algorithm.\n" + "\t(Value should be between 0 - 1, Default = 0.2).", "M", 1, "-M <momentum>"));
		newVector.addElement(new Option("\tNumber of epochs to train through.\n" + "\t(Default = 500).", "N", 1, "-N <number of epochs>"));
		newVector.addElement(new Option("\tPercentage size of validation set to use to terminate\n" + "\ttraining (if this is non zero it can pre-empt num of epochs.\n" + "\t(Value should be between 0 - 100, Default = 0).", "V", 1,
				"-V <percentage size of validation set>"));
		newVector.addElement(new Option("\tThe value used to seed the random number generator\n" + "\t(Value should be >= 0 and and a long, Default = 0).", "S", 1, "-S <seed>"));
		newVector.addElement(new Option("\tThe consequetive number of errors allowed for validation\n" + "\ttesting before the netwrok terminates.\n" + "\t(Value should be > 0, Default = 20).", "E", 1,
				"-E <threshold for number of consequetive errors>"));
		newVector.addElement(new Option("\tGUI will be opened.\n" + "\t(Use this to bring up a GUI).", "G", 0, "-G"));
		newVector.addElement(new Option("\tAutocreation of the network connections will NOT be done.\n" + "\t(This will be ignored if -G is NOT set)", "A", 0, "-A"));
		newVector.addElement(new Option("\tA NominalToBinary filter will NOT automatically be used.\n" + "\t(Set this to not use a NominalToBinary filter).", "B", 0, "-B"));
		newVector.addElement(new Option("\tThe hidden layers to be created for the network.\n" + "\t(Value should be a list of comma separated Natural \n" + "\tnumbers or the letters 'a' = (attribs + classes) / 2, \n"
				+ "\t'i' = attribs, 'o' = classes, 't' = attribs .+ classes)\n" + "\tfor wildcard values, Default = a).", "H", 1, "-H <comma seperated numbers for nodes on each layer>"));
		newVector.addElement(new Option("\tNormalizing a numeric class will NOT be done.\n" + "\t(Set this to not normalize the class if it's numeric).", "C", 0, "-C"));
		newVector.addElement(new Option("\tNormalizing the attributes will NOT be done.\n" + "\t(Set this to not normalize the attributes).", "I", 0, "-I"));
		newVector.addElement(new Option("\tReseting the network will NOT be allowed.\n" + "\t(Set this to not allow the network to reset).", "R", 0, "-R"));
		newVector.addElement(new Option("\tLearning rate decay will occur.\n" + "\t(Set this to cause the learning rate to decay).", "D", 0, "-D"));

		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 *
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 *
	 * <pre>
	 * -L &lt;learning rate&gt;
	 *  Learning Rate for the backpropagation algorithm.
	 *  (Value should be between 0 - 1, Default = 0.3).
	 * </pre>
	 *
	 * <pre>
	 * -M &lt;momentum&gt;
	 *  Momentum Rate for the backpropagation algorithm.
	 *  (Value should be between 0 - 1, Default = 0.2).
	 * </pre>
	 *
	 * <pre>
	 * -N &lt;number of epochs&gt;
	 *  Number of epochs to train through.
	 *  (Default = 500).
	 * </pre>
	 *
	 * <pre>
	 * -V &lt;percentage size of validation set&gt;
	 *  Percentage size of validation set to use to terminate
	 *  training (if this is non zero it can pre-empt num of epochs.
	 *  (Value should be between 0 - 100, Default = 0).
	 * </pre>
	 *
	 * <pre>
	 * -S &lt;seed&gt;
	 *  The value used to seed the random number generator
	 *  (Value should be &gt;= 0 and and a long, Default = 0).
	 * </pre>
	 *
	 * <pre>
	 * -E &lt;threshold for number of consequetive errors&gt;
	 *  The consequetive number of errors allowed for validation
	 *  testing before the netwrok terminates.
	 *  (Value should be &gt; 0, Default = 20).
	 * </pre>
	 *
	 * <pre>
	 * -G
	 *  GUI will be opened.
	 *  (Use this to bring up a GUI).
	 * </pre>
	 *
	 * <pre>
	 * -A
	 *  Autocreation of the network connections will NOT be done.
	 *  (This will be ignored if -G is NOT set)
	 * </pre>
	 *
	 * <pre>
	 * -B
	 *  A NominalToBinary filter will NOT automatically be used.
	 *  (Set this to not use a NominalToBinary filter).
	 * </pre>
	 *
	 * <pre>
	 * -H &lt;comma seperated numbers for nodes on each layer&gt;
	 *  The hidden layers to be created for the network.
	 *  (Value should be a list of comma separated Natural
	 *  numbers or the letters 'a' = (attribs + classes) / 2,
	 *  'i' = attribs, 'o' = classes, 't' = attribs .+ classes)
	 *  for wildcard values, Default = a).
	 * </pre>
	 *
	 * <pre>
	 * -C
	 *  Normalizing a numeric class will NOT be done.
	 *  (Set this to not normalize the class if it's numeric).
	 * </pre>
	 *
	 * <pre>
	 * -I
	 *  Normalizing the attributes will NOT be done.
	 *  (Set this to not normalize the attributes).
	 * </pre>
	 *
	 * <pre>
	 * -R
	 *  Reseting the network will NOT be allowed.
	 *  (Set this to not allow the network to reset).
	 * </pre>
	 *
	 * <pre>
	 * -D
	 *  Learning rate decay will occur.
	 *  (Set this to cause the learning rate to decay).
	 * </pre>
	 *
	 * <!-- options-end -->
	 *
	 * @param options
	 *          the list of options as an array of strings
	 * @throws Exception
	 *           if an option is not supported
	 */
	@Override
	public void setOptions(final String[] options) throws Exception {
		// the defaults can be found here!!!!
		String learningString = Utils.getOption('L', options);
		if (learningString.length() != 0) {
			this.setLearningRate((new Double(learningString)).doubleValue());
		} else {
			this.setLearningRate(0.3);
		}
		String momentumString = Utils.getOption('M', options);
		if (momentumString.length() != 0) {
			this.setMomentum((new Double(momentumString)).doubleValue());
		} else {
			this.setMomentum(0.2);
		}
		String epochsString = Utils.getOption('N', options);
		if (epochsString.length() != 0) {
			this.setTrainingTime(Integer.parseInt(epochsString));
		} else {
			this.setTrainingTime(500);
		}
		String valSizeString = Utils.getOption('V', options);
		if (valSizeString.length() != 0) {
			this.setValidationSetSize(Integer.parseInt(valSizeString));
		} else {
			this.setValidationSetSize(0);
		}
		String seedString = Utils.getOption('S', options);
		if (seedString.length() != 0) {
			this.setSeed(Integer.parseInt(seedString));
		} else {
			this.setSeed(0);
		}
		String thresholdString = Utils.getOption('E', options);
		if (thresholdString.length() != 0) {
			this.setValidationThreshold(Integer.parseInt(thresholdString));
		} else {
			this.setValidationThreshold(20);
		}
		String hiddenLayers = Utils.getOption('H', options);
		if (hiddenLayers.length() != 0) {
			this.setHiddenLayers(hiddenLayers);
		} else {
			this.setHiddenLayers("a");
		}
		if (Utils.getFlag('G', options)) {
			this.setGUI(true);
		} else {
			this.setGUI(false);
		} // small note. since the gui is the only option that can change the other
			// options this should be set first to allow the other options to set
			// properly
		if (Utils.getFlag('A', options)) {
			this.setAutoBuild(false);
		} else {
			this.setAutoBuild(true);
		}
		if (Utils.getFlag('B', options)) {
			this.setNominalToBinaryFilter(false);
		} else {
			this.setNominalToBinaryFilter(true);
		}
		if (Utils.getFlag('C', options)) {
			this.setNormalizeNumericClass(false);
		} else {
			this.setNormalizeNumericClass(true);
		}
		if (Utils.getFlag('I', options)) {
			this.setNormalizeAttributes(false);
		} else {
			this.setNormalizeAttributes(true);
		}
		if (Utils.getFlag('R', options)) {
			this.setReset(false);
		} else {
			this.setReset(true);
		}
		if (Utils.getFlag('D', options)) {
			this.setDecay(true);
		} else {
			this.setDecay(false);
		}

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of NeuralNet.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	@Override
	public String[] getOptions() {

		Vector<String> options = new Vector<>();

		options.add("-L");
		options.add("" + this.getLearningRate());
		options.add("-M");
		options.add("" + this.getMomentum());
		options.add("-N");
		options.add("" + this.getTrainingTime());
		options.add("-V");
		options.add("" + this.getValidationSetSize());
		options.add("-S");
		options.add("" + this.getSeed());
		options.add("-E");
		options.add("" + this.getValidationThreshold());
		options.add("-H");
		options.add(this.getHiddenLayers());
		if (this.getGUI()) {
			options.add("-G");
		}
		if (!this.getAutoBuild()) {
			options.add("-A");
		}
		if (!this.getNominalToBinaryFilter()) {
			options.add("-B");
		}
		if (!this.getNormalizeNumericClass()) {
			options.add("-C");
		}
		if (!this.getNormalizeAttributes()) {
			options.add("-I");
		}
		if (!this.getReset()) {
			options.add("-R");
		}
		if (this.getDecay()) {
			options.add("-D");
		}

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	}

	/**
	 * @return string describing the model.
	 */
	@Override
	public String toString() {
		// only ZeroR model?
		if (this.m_useDefaultModel) {
			StringBuffer buf = new StringBuffer();
			buf.append(this.getClass().getName().replaceAll(".*\\.", "") + "\n");
			buf.append(this.getClass().getName().replaceAll(".*\\.", "").replaceAll(".", "=") + "\n\n");
			buf.append("Warning: No model could be built, hence ZeroR model is used:\n\n");
			buf.append(this.m_ZeroR.toString());
			return buf.toString();
		}

		StringBuffer model = new StringBuffer(this.m_neuralNodes.length * 100);
		// just a rough size guess
		NeuralNode con;
		double[] weights;
		NeuralConnection[] inputs;
		for (NeuralConnection m_neuralNode : this.m_neuralNodes) {
			con = (NeuralNode) m_neuralNode; // this would need a change
												// for items other than nodes!!!
			weights = con.getWeights();
			inputs = con.getInputs();
			if (con.getMethod() instanceof SigmoidUnit) {
				model.append("Sigmoid ");
			} else if (con.getMethod() instanceof LinearUnit) {
				model.append("Linear ");
			}
			model.append("Node " + con.getId() + "\n    Inputs    Weights\n");
			model.append("    Threshold    " + weights[0] + "\n");
			for (int nob = 1; nob < con.getNumInputs() + 1; nob++) {
				if ((inputs[nob - 1].getType() & NeuralConnection.PURE_INPUT) == NeuralConnection.PURE_INPUT) {
					model.append("    Attrib " + this.m_instances.attribute(((NeuralEnd) inputs[nob - 1]).getLink()).name() + "    " + weights[nob] + "\n");
				} else {
					model.append("    Node " + inputs[nob - 1].getId() + "    " + weights[nob] + "\n");
				}
			}
		}
		// now put in the ends
		for (NeuralEnd m_output : this.m_outputs) {
			inputs = m_output.getInputs();
			model.append("Class " + this.m_instances.classAttribute().value(m_output.getLink()) + "\n    Input\n");
			for (int nob = 0; nob < m_output.getNumInputs(); nob++) {
				if ((inputs[nob].getType() & NeuralConnection.PURE_INPUT) == NeuralConnection.PURE_INPUT) {
					model.append("    Attrib " + this.m_instances.attribute(((NeuralEnd) inputs[nob]).getLink()).name() + "\n");
				} else {
					model.append("    Node " + inputs[nob].getId() + "\n");
				}
			}
		}
		return model.toString();
	}

	/**
	 * This will return a string describing the classifier.
	 *
	 * @return The string.
	 */
	public String globalInfo() {
		return "A Classifier that uses backpropagation to classify instances.\n" + "This network can be built by hand, created by an algorithm or both. " + "The network can also be monitored and modified during training time. "
				+ "The nodes in this network are all sigmoid (except for when the class " + "is numeric in which case the the output nodes become unthresholded " + "linear units).";
	}

	/**
	 * @return a string to describe the learning rate option.
	 */
	public String learningRateTipText() {
		return "The amount the" + " weights are updated.";
	}

	/**
	 * @return a string to describe the momentum option.
	 */
	public String momentumTipText() {
		return "Momentum applied to the weights during updating.";
	}

	/**
	 * @return a string to describe the AutoBuild option.
	 */
	public String autoBuildTipText() {
		return "Adds and connects up hidden layers in the network.";
	}

	/**
	 * @return a string to describe the random seed option.
	 */
	public String seedTipText() {
		return "Seed used to initialise the random number generator." + "Random numbers are used for setting the initial weights of the" + " connections betweem nodes, and also for shuffling the training data.";
	}

	/**
	 * @return a string to describe the validation threshold option.
	 */
	public String validationThresholdTipText() {
		return "Used to terminate validation testing." + "The value here dictates how many times in a row the validation set" + " error can get worse before training is terminated.";
	}

	/**
	 * @return a string to describe the GUI option.
	 */
	public String GUITipText() {
		return "Brings up a gui interface." + " This will allow the pausing and altering of the nueral network" + " during training.\n\n" + "* To add a node left click (this node will be automatically selected,"
				+ " ensure no other nodes were selected).\n" + "* To select a node left click on it either while no other node is" + " selected or while holding down the control key (this toggles that"
				+ " node as being selected and not selected.\n" + "* To connect a node, first have the start node(s) selected, then click" + " either the end node or on an empty space (this will create a new node"
				+ " that is connected with the selected nodes). The selection status of" + " nodes will stay the same after the connection. (Note these are" + " directed connections, also a connection between two nodes will not"
				+ " be established more than once and certain connections that are" + " deemed to be invalid will not be made).\n" + "* To remove a connection select one of the connected node(s) in the"
				+ " connection and then right click the other node (it does not matter" + " whether the node is the start or end the connection will be removed" + ").\n"
				+ "* To remove a node right click it while no other nodes (including it)" + " are selected. (This will also remove all connections to it)\n." + "* To deselect a node either left click it while holding down control,"
				+ " or right click on empty space.\n" + "* The raw inputs are provided from the labels on the left.\n" + "* The red nodes are hidden layers.\n" + "* The orange nodes are the output nodes.\n"
				+ "* The labels on the right show the class the output node represents." + " Note that with a numeric class the output node will automatically be" + " made into an unthresholded linear unit.\n\n"
				+ "Alterations to the neural network can only be done while the network" + " is not running, This also applies to the learning rate and other" + " fields on the control panel.\n\n"
				+ "* You can accept the network as being finished at any time.\n" + "* The network is automatically paused at the beginning.\n" + "* There is a running indication of what epoch the network is up to"
				+ " and what the (rough) error for that epoch was (or for" + " the validation if that is being used). Note that this error value" + " is based on a network that changes as the value is computed."
				+ " (also depending on whether" + " the class is normalized will effect the error reported for numeric" + " classes.\n" + "* Once the network is done it will pause again and either wait to be"
				+ " accepted or trained more.\n\n" + "Note that if the gui is not set the network will not require any" + " interaction.\n";
	}

	/**
	 * @return a string to describe the validation size option.
	 */
	public String validationSetSizeTipText() {
		return "The percentage size of the validation set." + "(The training will continue until it is observed that" + " the error on the validation set has been consistently getting" + " worse, or if the training time is reached).\n"
				+ "If This is set to zero no validation set will be used and instead" + " the network will train for the specified number of epochs.";
	}

	/**
	 * @return a string to describe the learning rate option.
	 */
	public String trainingTimeTipText() {
		return "The number of epochs to train through." + " If the validation set is non-zero then it can terminate the network" + " early";
	}

	/**
	 * @return a string to describe the nominal to binary option.
	 */
	public String nominalToBinaryFilterTipText() {
		return "This will preprocess the instances with the filter." + " This could help improve performance if there are nominal attributes" + " in the data.";
	}

	/**
	 * @return a string to describe the hidden layers in the network.
	 */
	public String hiddenLayersTipText() {
		return "This defines the hidden layers of the neural network." + " This is a list of positive whole numbers. 1 for each hidden layer." + " Comma seperated. To have no hidden layers put a single 0 here."
				+ " This will only be used if autobuild is set. There are also wildcard" + " values 'a' = (attribs + classes) / 2, 'i' = attribs, 'o' = classes" + " , 't' = attribs + classes.";
	}

	/**
	 * @return a string to describe the nominal to binary option.
	 */
	public String normalizeNumericClassTipText() {
		return "This will normalize the class if it's numeric." + " This could help improve performance of the network, It normalizes" + " the class to be between -1 and 1. Note that this is only internally"
				+ ", the output will be scaled back to the original range.";
	}

	/**
	 * @return a string to describe the nominal to binary option.
	 */
	public String normalizeAttributesTipText() {
		return "This will normalize the attributes." + " This could help improve performance of the network." + " This is not reliant on the class being numeric. This will also"
				+ " normalize nominal attributes as well (after they have been run" + " through the nominal to binary filter if that is in use) so that the" + " nominal values are between -1 and 1";
	}

	/**
	 * @return a string to describe the Reset option.
	 */
	public String resetTipText() {
		return "This will allow the network to reset with a lower learning rate." + " If the network diverges from the answer this will automatically" + " reset the network with a lower learning rate and begin training"
				+ " again. This option is only available if the gui is not set. Note" + " that if the network diverges but isn't allowed to reset it will" + " fail the training process and return an error message.";
	}

	/**
	 * @return a string to describe the Decay option.
	 */
	public String decayTipText() {
		return "This will cause the learning rate to decrease." + " This will divide the starting learning rate by the epoch number, to" + " determine what the current learning rate should be. This may help"
				+ " to stop the network from diverging from the target output, as well" + " as improve general performance. Note that the decaying learning" + " rate will not be shown in the gui, only the original learning rate"
				+ ". If the learning rate is changed in the gui, this is treated as the" + " starting learning rate.";
	}

	/**
	 * Returns the revision string.
	 *
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision$");
	}
}
