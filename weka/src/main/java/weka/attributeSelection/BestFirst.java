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
 *    BestFirst.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.attributeSelection;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;

import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;

/**
 <!-- globalinfo-start -->
 * BestFirst:<br/>
 * <br/>
 * Searches the space of attribute subsets by greedy hillclimbing augmented with
 * a backtracking facility. Setting the number of consecutive non-improving
 * nodes allowed controls the level of backtracking done. Best first may start
 * with the empty set of attributes and search forward, or start with the full
 * set of attributes and search backward, or start at any point and search in
 * both directions (by considering all possible single attribute additions and
 * deletions at a given point).<br/>
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -P &lt;start set&gt;
 *  Specify a starting set of attributes.
 *  Eg. 1,3,5-7.
 * </pre>
 *
 * <pre>
 * -D &lt;0 = backward | 1 = forward | 2 = bi-directional&gt;
 *  Direction of search. (default = 1).
 * </pre>
 *
 * <pre>
 * -N &lt;num&gt;
 *  Number of non-improving nodes to
 *  consider before terminating search.
 * </pre>
 *
 * <pre>
 * -S &lt;num&gt;
 *  Size of lookup cache for evaluated subsets.
 *  Expressed as a multiple of the number of
 *  attributes in the data set. (default = 1)
 * </pre>
 *
 <!-- options-end -->
 *
 * @author Mark Hall (mhall@cs.waikato.ac.nz) Martin Guetlein (cashing merit of
 *         expanded nodes)
 * @version $Revision$
 */
public class BestFirst extends ASSearch implements OptionHandler, StartSetHandler {

	/** for serialization */
	static final long serialVersionUID = 7841338689536821867L;

	// Inner classes
	/**
	 * Class for a node in a linked list. Used in best first search.
	 *
	 * @author Mark Hall (mhall@cs.waikato.ac.nz)
	 **/
	public class Link2 implements Serializable, RevisionHandler {

		/** for serialization */
		static final long serialVersionUID = -8236598311516351420L;

		/* BitSet group; */
		Object[] m_data;
		double m_merit;

		/**
		 * Constructor
		 */
		public Link2(final Object[] data, final double mer) {
			// group = (BitSet)gr.clone();
			this.m_data = data;
			this.m_merit = mer;
		}

		/** Get a group */
		public Object[] getData() {
			return this.m_data;
		}

		@Override
		public String toString() {
			return ("Node: " + this.m_data.toString() + "  " + this.m_merit);
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
	 * Class for handling a linked list. Used in best first search. Extends the
	 * Vector class.
	 *
	 * @author Mark Hall (mhall@cs.waikato.ac.nz)
	 **/
	public class LinkedList2 extends ArrayList<Link2> {

		/** for serialization */
		static final long serialVersionUID = 3250538292330398929L;

		/** Max number of elements in the list */
		int m_MaxSize;

		// ================
		// Public methods
		// ================
		public LinkedList2(final int sz) {
			super();
			this.m_MaxSize = sz;
		}

		/**
		 * removes an element (Link) at a specific index from the list.
		 *
		 * @param index the index of the element to be removed.
		 **/
		public void removeLinkAt(final int index) throws Exception {

			if ((index >= 0) && (index < this.size())) {
				this.remove(index);
			} else {
				throw new Exception("index out of range (removeLinkAt)");
			}
		}

		/**
		 * returns the element (Link) at a specific index from the list.
		 *
		 * @param index the index of the element to be returned.
		 **/
		public Link2 getLinkAt(final int index) throws Exception {

			if (this.size() == 0) {
				throw new Exception("List is empty (getLinkAt)");
			} else {
				if ((index >= 0) && (index < this.size())) {
					return ((this.get(index)));
				} else {
					throw new Exception("index out of range (getLinkAt)");
				}
			}
		}

		/**
		 * adds an element (Link) to the list.
		 *
		 * @param data the attribute set specification
		 * @param mer the "merit" of this attribute set
		 **/
		public void addToList(final Object[] data, final double mer) throws Exception {
			Link2 newL = new Link2(data, mer);

			if (this.size() == 0) {
				this.add(newL);
			} else {
				if (mer > (this.get(0)).m_merit) {
					if (this.size() == this.m_MaxSize) {
						this.removeLinkAt(this.m_MaxSize - 1);
					}

					// ----------
					this.add(0, newL);
				} else {
					int i = 0;
					int size = this.size();
					boolean done = false;

					// ------------
					// don't insert if list contains max elements an this
					// is worst than the last
					if ((size == this.m_MaxSize) && (mer <= this.get(this.size() - 1).m_merit)) {

					}
					// ---------------
					else {
						while ((!done) && (i < size)) {
							if (mer > (this.get(i)).m_merit) {
								if (size == this.m_MaxSize) {
									this.removeLinkAt(this.m_MaxSize - 1);
								}

								// ---------------------
								this.add(i, newL);
								done = true;
							} else {
								if (i == size - 1) {
									this.add(newL);
									done = true;
								} else {
									i++;
								}
							}
						}
					}
				}
			}
		}

		/**
		 * Returns the revision string.
		 *
		 * @return the revision
		 */
		public String getRevision() {
			return RevisionUtils.extract("$Revision$");
		}
	}

	// member variables
	/** maximum number of stale nodes before terminating search */
	protected int m_maxStale;

	/** 0 == backward search, 1 == forward search, 2 == bidirectional */
	protected int m_searchDirection;

	/** search direction: backward */
	protected static final int SELECTION_BACKWARD = 0;
	/** search direction: forward */
	protected static final int SELECTION_FORWARD = 1;
	/** search direction: bidirectional */
	protected static final int SELECTION_BIDIRECTIONAL = 2;
	/** search directions */
	public static final Tag[] TAGS_SELECTION = { new Tag(SELECTION_BACKWARD, "Backward"), new Tag(SELECTION_FORWARD, "Forward"), new Tag(SELECTION_BIDIRECTIONAL, "Bi-directional"), };

	/** holds an array of starting attributes */
	protected int[] m_starting;

	/** holds the start set for the search as a Range */
	protected Range m_startRange;

	/** does the data have a class */
	protected boolean m_hasClass;

	/** holds the class index */
	protected int m_classIndex;

	/** number of attributes in the data */
	protected int m_numAttribs;

	/** total number of subsets evaluated during a search */
	protected int m_totalEvals;

	/** for debugging */
	protected boolean m_debug;

	/** holds the merit of the best subset found */
	protected double m_bestMerit;

	/** holds the maximum size of the lookup cache for evaluated subsets */
	protected int m_cacheSize;

	/**
	 * Returns a string describing this search method
	 *
	 * @return a description of the search method suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "BestFirst:\n\n" + "Searches the space of attribute subsets by greedy hillclimbing " + "augmented with a backtracking facility. Setting the number of " + "consecutive non-improving nodes allowed controls the level of "
				+ "backtracking done. Best first may start with the empty set of " + "attributes and search forward, or start with the full set of " + "attributes and search backward, or start at any point and search "
				+ "in both directions (by considering all possible single attribute " + "additions and deletions at a given point).\n";
	}

	/**
	 * Constructor
	 */
	public BestFirst() {
		this.resetOptions();
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 *
	 **/
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(4);

		newVector.addElement(new Option("\tSpecify a starting set of attributes." + "\n\tEg. 1,3,5-7.", "P", 1, "-P <start set>"));
		newVector.addElement(new Option("\tDirection of search. (default = 1).", "D", 1, "-D <0 = backward | 1 = forward " + "| 2 = bi-directional>"));
		newVector.addElement(new Option("\tNumber of non-improving nodes to" + "\n\tconsider before terminating search.", "N", 1, "-N <num>"));
		newVector.addElement(new Option("\tSize of lookup cache for evaluated subsets." + "\n\tExpressed as a multiple of the number of" + "\n\tattributes in the data set. (default = 1)", "S", 1, "-S <num>"));

		return newVector.elements();
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 *
	 <!-- options-start -->
	 * Valid options are:
	 * <p/>
	 *
	 * <pre>
	 * -P &lt;start set&gt;
	 *  Specify a starting set of attributes.
	 *  Eg. 1,3,5-7.
	 * </pre>
	 *
	 * <pre>
	 * -D &lt;0 = backward | 1 = forward | 2 = bi-directional&gt;
	 *  Direction of search. (default = 1).
	 * </pre>
	 *
	 * <pre>
	 * -N &lt;num&gt;
	 *  Number of non-improving nodes to
	 *  consider before terminating search.
	 * </pre>
	 *
	 * <pre>
	 * -S &lt;num&gt;
	 *  Size of lookup cache for evaluated subsets.
	 *  Expressed as a multiple of the number of
	 *  attributes in the data set. (default = 1)
	 * </pre>
	 *
	 <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 *
	 **/
	@Override
	public void setOptions(final String[] options) throws Exception {
		String optionString;
		this.resetOptions();

		optionString = Utils.getOption('P', options);
		if (optionString.length() != 0) {
			this.setStartSet(optionString);
		}

		optionString = Utils.getOption('D', options);

		if (optionString.length() != 0) {
			this.setDirection(new SelectedTag(Integer.parseInt(optionString), TAGS_SELECTION));
		} else {
			this.setDirection(new SelectedTag(SELECTION_FORWARD, TAGS_SELECTION));
		}

		optionString = Utils.getOption('N', options);

		if (optionString.length() != 0) {
			this.setSearchTermination(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('S', options);
		if (optionString.length() != 0) {
			this.setLookupCacheSize(Integer.parseInt(optionString));
		}

		this.m_debug = Utils.getFlag('Z', options);
	}

	/**
	 * Set the maximum size of the evaluated subset cache (hashtable). This is
	 * expressed as a multiplier for the number of attributes in the data set.
	 * (default = 1).
	 *
	 * @param size the maximum size of the hashtable
	 */
	public void setLookupCacheSize(final int size) {
		if (size >= 0) {
			this.m_cacheSize = size;
		}
	}

	/**
	 * Return the maximum size of the evaluated subset cache (expressed as a
	 * multiplier for the number of attributes in a data set.
	 *
	 * @return the maximum size of the hashtable.
	 */
	public int getLookupCacheSize() {
		return this.m_cacheSize;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String lookupCacheSizeTipText() {
		return "Set the maximum size of the lookup cache of evaluated subsets. This is " + "expressed as a multiplier of the number of attributes in the data set. " + "(default = 1).";
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String startSetTipText() {
		return "Set the start point for the search. This is specified as a comma " + "seperated list off attribute indexes starting at 1. It can include " + "ranges. Eg. 1,2,5-9,17.";
	}

	/**
	 * Sets a starting set of attributes for the search. It is the search method's
	 * responsibility to report this start set (if any) in its toString() method.
	 *
	 * @param startSet a string containing a list of attributes (and or ranges),
	 *          eg. 1,2,6,10-15.
	 * @throws Exception if start set can't be set.
	 */
	@Override
	public void setStartSet(final String startSet) throws Exception {
		this.m_startRange.setRanges(startSet);
	}

	/**
	 * Returns a list of attributes (and or attribute ranges) as a String
	 *
	 * @return a list of attributes (and or attribute ranges)
	 */
	@Override
	public String getStartSet() {
		return this.m_startRange.getRanges();
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String searchTerminationTipText() {
		return "Specify the number of consecutive non-improving nodes to allow " + "before terminating the search.";
	}

	/**
	 * Set the numnber of non-improving nodes to consider before terminating
	 * search.
	 *
	 * @param t the number of non-improving nodes
	 * @throws Exception if t is less than 1
	 */
	public void setSearchTermination(final int t) throws Exception {
		if (t < 1) {
			throw new Exception("Value of -N must be > 0.");
		}

		this.m_maxStale = t;
	}

	/**
	 * Get the termination criterion (number of non-improving nodes).
	 *
	 * @return the number of non-improving nodes
	 */
	public int getSearchTermination() {
		return this.m_maxStale;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String directionTipText() {
		return "Set the direction of the search.";
	}

	/**
	 * Set the search direction
	 *
	 * @param d the direction of the search
	 */
	public void setDirection(final SelectedTag d) {

		if (d.getTags() == TAGS_SELECTION) {
			this.m_searchDirection = d.getSelectedTag().getID();
		}
	}

	/**
	 * Get the search direction
	 *
	 * @return the direction of the search
	 */
	public SelectedTag getDirection() {

		return new SelectedTag(this.m_searchDirection, TAGS_SELECTION);
	}

	/**
	 * Gets the current settings of BestFirst.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	@Override
	public String[] getOptions() {

		Vector<String> options = new Vector<String>();

		if (!(this.getStartSet().equals(""))) {
			options.add("-P");
			options.add("" + this.startSetToString());
		}
		options.add("-D");
		options.add("" + this.m_searchDirection);
		options.add("-N");
		options.add("" + this.m_maxStale);

		return options.toArray(new String[0]);
	}

	/**
	 * converts the array of starting attributes to a string. This is used by
	 * getOptions to return the actual attributes specified as the starting set.
	 * This is better than using m_startRanges.getRanges() as the same start set
	 * can be specified in different ways from the command line---eg 1,2,3 == 1-3.
	 * This is to ensure that stuff that is stored in a database is comparable.
	 *
	 * @return a comma seperated list of individual attribute numbers as a String
	 */
	private String startSetToString() {
		StringBuffer FString = new StringBuffer();
		boolean didPrint;

		if (this.m_starting == null) {
			return this.getStartSet();
		}
		for (int i = 0; i < this.m_starting.length; i++) {
			didPrint = false;

			if ((this.m_hasClass == false) || (this.m_hasClass == true && i != this.m_classIndex)) {
				FString.append((this.m_starting[i] + 1));
				didPrint = true;
			}

			if (i == (this.m_starting.length - 1)) {
				FString.append("");
			} else {
				if (didPrint) {
					FString.append(",");
				}
			}
		}

		return FString.toString();
	}

	/**
	 * returns a description of the search as a String
	 *
	 * @return a description of the search
	 */
	@Override
	public String toString() {
		StringBuffer BfString = new StringBuffer();
		BfString.append("\tBest first.\n\tStart set: ");

		if (this.m_starting == null) {
			BfString.append("no attributes\n");
		} else {
			BfString.append(this.startSetToString() + "\n");
		}

		BfString.append("\tSearch direction: ");

		if (this.m_searchDirection == SELECTION_BACKWARD) {
			BfString.append("backward\n");
		} else {
			if (this.m_searchDirection == SELECTION_FORWARD) {
				BfString.append("forward\n");
			} else {
				BfString.append("bi-directional\n");
			}
		}

		BfString.append("\tStale search after " + this.m_maxStale + " node expansions\n");
		BfString.append("\tTotal number of subsets evaluated: " + this.m_totalEvals + "\n");
		BfString.append("\tMerit of best subset found: " + Utils.doubleToString(Math.abs(this.m_bestMerit), 8, 3) + "\n");
		return BfString.toString();
	}

	protected void printGroup(final BitSet tt, final int numAttribs) {
		int i;

		for (i = 0; i < numAttribs; i++) {
			if (tt.get(i) == true) {
				System.out.print((i + 1) + " ");
			}
		}

		System.out.println();
	}

	/**
	 * Searches the attribute subset space by best first search
	 *
	 * @param ASEval the attribute evaluator to guide the search
	 * @param data the training instances.
	 * @return an array (not necessarily ordered) of selected attribute indexes
	 * @throws Exception if the search can't be completed
	 */
	@Override
	public int[] search(final ASEvaluation ASEval, final Instances data) throws Exception {
		this.m_totalEvals = 0;
		if (!(ASEval instanceof SubsetEvaluator)) {
			throw new Exception(ASEval.getClass().getName() + " is not a " + "Subset evaluator!");
		}

		if (ASEval instanceof UnsupervisedSubsetEvaluator) {
			this.m_hasClass = false;
		} else {
			this.m_hasClass = true;
			this.m_classIndex = data.classIndex();
		}

		SubsetEvaluator ASEvaluator = (SubsetEvaluator) ASEval;
		this.m_numAttribs = data.numAttributes();
		int i, j;
		int best_size = 0;
		int size = 0;
		int done;
		int sd = this.m_searchDirection;
		BitSet best_group, temp_group;
		int stale;
		double best_merit;
		double merit;
		boolean z;
		boolean added;
		Link2 tl;
		Hashtable<String, Double> lookup = new Hashtable<String, Double>(this.m_cacheSize * this.m_numAttribs);
		int insertCount = 0;
		LinkedList2 bfList = new LinkedList2(this.m_maxStale);
		best_merit = -Double.MAX_VALUE;
		stale = 0;
		best_group = new BitSet(this.m_numAttribs);

		this.m_startRange.setUpper(this.m_numAttribs - 1);
		if (!(this.getStartSet().equals(""))) {
			this.m_starting = this.m_startRange.getSelection();
		}
		// If a starting subset has been supplied, then initialise the bitset
		if (this.m_starting != null) {
			for (i = 0; i < this.m_starting.length; i++) {
				if ((this.m_starting[i]) != this.m_classIndex) {
					best_group.set(this.m_starting[i]);
				}
			}

			best_size = this.m_starting.length;
			this.m_totalEvals++;
		} else {
			if (this.m_searchDirection == SELECTION_BACKWARD) {
				this.setStartSet("1-last");
				this.m_starting = new int[this.m_numAttribs];

				// init initial subset to all attributes
				for (i = 0, j = 0; i < this.m_numAttribs; i++) {
					// XXX thread interrupted; throw exception
					if (Thread.interrupted()) {
						throw new InterruptedException("Killed WEKA");
					}
					if (i != this.m_classIndex) {
						best_group.set(i);
						this.m_starting[j++] = i;
					}
				}

				best_size = this.m_numAttribs - 1;
				this.m_totalEvals++;
			}
		}

		// evaluate the initial subset
		best_merit = ASEvaluator.evaluateSubset(best_group);
		// add the initial group to the list and the hash table
		Object[] best = new Object[1];
		best[0] = best_group.clone();
		bfList.addToList(best, best_merit);
		BitSet tt = (BitSet) best_group.clone();
		String hashC = tt.toString();
		lookup.put(hashC, new Double(best_merit));

		while (stale < this.m_maxStale) {
			// XXX thread interrupted; throw exception
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA");
			}
			added = false;

			if (this.m_searchDirection == SELECTION_BIDIRECTIONAL) {
				// bi-directional search
				done = 2;
				sd = SELECTION_FORWARD;
			} else {
				done = 1;
			}

			// finished search?
			if (bfList.size() == 0) {
				stale = this.m_maxStale;
				break;
			}

			// copy the attribute set at the head of the list
			tl = bfList.getLinkAt(0);
			temp_group = (BitSet) (tl.getData()[0]);
			temp_group = (BitSet) temp_group.clone();
			// remove the head of the list
			bfList.removeLinkAt(0);
			// count the number of bits set (attributes)
			int kk;

			for (kk = 0, size = 0; kk < this.m_numAttribs; kk++) {
				if (temp_group.get(kk)) {
					size++;
				}
			}

			do {
				// XXX thread interrupted; throw exception
				if (Thread.interrupted()) {
					throw new InterruptedException("Killed WEKA");
				}
				for (i = 0; i < this.m_numAttribs; i++) {
					if (sd == SELECTION_FORWARD) {
						z = ((i != this.m_classIndex) && (!temp_group.get(i)));
					} else {
						z = ((i != this.m_classIndex) && (temp_group.get(i)));
					}

					if (z) {
						// set the bit (attribute to add/delete)
						if (sd == SELECTION_FORWARD) {
							temp_group.set(i);
							size++;
						} else {
							temp_group.clear(i);
							size--;
						}

						/*
						 * if this subset has been seen before, then it is already in the
						 * list (or has been fully expanded)
						 */
						tt = (BitSet) temp_group.clone();
						hashC = tt.toString();

						if (lookup.containsKey(hashC) == false) {
							merit = ASEvaluator.evaluateSubset(temp_group);
							this.m_totalEvals++;

							// insert this one in the hashtable
							if (insertCount > this.m_cacheSize * this.m_numAttribs) {
								lookup = new Hashtable<String, Double>(this.m_cacheSize * this.m_numAttribs);
								insertCount = 0;
							}
							hashC = tt.toString();
							lookup.put(hashC, new Double(merit));
							insertCount++;
						} else {
							merit = lookup.get(hashC).doubleValue();
						}

						// insert this one in the list
						Object[] add = new Object[1];
						add[0] = tt.clone();
						bfList.addToList(add, merit);

						if (this.m_debug) {
							System.out.print("Group: ");
							this.printGroup(tt, this.m_numAttribs);
							System.out.println("Merit: " + merit);
						}

						// is this better than the best?
						if (sd == SELECTION_FORWARD) {
							z = ((merit - best_merit) > 0.00001);
						} else {
							if (merit == best_merit) {
								z = (size < best_size);
							} else {
								z = (merit > best_merit);
							}
						}

						if (z) {
							added = true;
							stale = 0;
							best_merit = merit;
							// best_size = (size + best_size);
							best_size = size;
							best_group = (BitSet) (temp_group.clone());
						}

						// unset this addition(deletion)
						if (sd == SELECTION_FORWARD) {
							temp_group.clear(i);
							size--;
						} else {
							temp_group.set(i);
							size++;
						}
					}
				}

				if (done == 2) {
					sd = SELECTION_BACKWARD;
				}

				done--;
			} while (done > 0);

			/*
			 * if we haven't added a new attribute subset then full expansion of this
			 * node hasen't resulted in anything better
			 */
			if (!added) {
				stale++;
			}
		}

		this.m_bestMerit = best_merit;
		return this.attributeList(best_group);
	}

	/**
	 * Reset options to default values
	 */
	protected void resetOptions() {
		this.m_maxStale = 5;
		this.m_searchDirection = SELECTION_FORWARD;
		this.m_starting = null;
		this.m_startRange = new Range();
		this.m_classIndex = -1;
		this.m_totalEvals = 0;
		this.m_cacheSize = 1;
		this.m_debug = false;
	}

	/**
	 * converts a BitSet into a list of attribute indexes
	 *
	 * @param group the BitSet to convert
	 * @return an array of attribute indexes
	 **/
	protected int[] attributeList(final BitSet group) {
		int count = 0;

		// count how many were selected
		for (int i = 0; i < this.m_numAttribs; i++) {
			if (group.get(i)) {
				count++;
			}
		}

		int[] list = new int[count];
		count = 0;

		for (int i = 0; i < this.m_numAttribs; i++) {
			if (group.get(i)) {
				list[count++] = i;
			}
		}

		return list;
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
