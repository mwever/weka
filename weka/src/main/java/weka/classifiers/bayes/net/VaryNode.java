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
 * VaryNode.java
 * Copyright (C) 2002-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.bayes.net;

import java.io.Serializable;

import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

/**
 * Part of ADTree implementation. See ADNode.java for more details.
 *
 * @author Remco Bouckaert (rrb@xm.co.nz)
 * @version $Revision$
 */
public class VaryNode implements Serializable, RevisionHandler {

	/** for serialization */
	private static final long serialVersionUID = -6196294370675872424L;

	/** index of the node varied **/
	public int m_iNode;

	/** most common value **/
	public int m_nMCV;

	/** list of ADNode children **/
	public ADNode[] m_ADNodes;

	/** Creates new VaryNode */
	public VaryNode(final int iNode) {
		this.m_iNode = iNode;
	}

	/**
	 * get counts for specific instantiation of a set of nodes
	 * 
	 * @param nCounts array for storing counts
	 * @param nNodes array of node indexes
	 * @param nOffsets offset for nodes in nNodes in nCounts
	 * @param iNode index into nNode indicating current node
	 * @param iOffset Offset into nCounts due to nodes below iNode
	 * @param parent parant ADNode of this VaryNode
	 * @param bSubstract indicate whether counts should be added or substracted
	* @throws InterruptedException 
	 */
	public void getCounts(final int[] nCounts, final int[] nNodes, final int[] nOffsets, final int iNode, final int iOffset, final ADNode parent, final boolean bSubstract) throws InterruptedException {
		for (int iValue = 0; iValue < this.m_ADNodes.length; iValue++) {
			if (iValue != this.m_nMCV) {
				if (this.m_ADNodes[iValue] != null) {
					this.m_ADNodes[iValue].getCounts(nCounts, nNodes, nOffsets, iNode + 1, iOffset + nOffsets[iNode] * iValue, bSubstract);
				}
			} else {
				parent.getCounts(nCounts, nNodes, nOffsets, iNode + 1, iOffset + nOffsets[iNode] * iValue, bSubstract);
				for (int iValue2 = 0; iValue2 < this.m_ADNodes.length; iValue2++) {
					if (iValue2 != this.m_nMCV && this.m_ADNodes[iValue2] != null) {
						this.m_ADNodes[iValue2].getCounts(nCounts, nNodes, nOffsets, iNode + 1, iOffset + nOffsets[iNode] * iValue, !bSubstract);
					}
				}
			}
		}
	}

	/**
	 * print is used for debugging only, called from ADNode
	 * 
	 * @param sTab amount of space.
	 */
	public void print(final String sTab) {
		for (int iValue = 0; iValue < this.m_ADNodes.length; iValue++) {
			System.out.print(sTab + iValue + ": ");
			if (this.m_ADNodes[iValue] == null) {
				if (iValue == this.m_nMCV) {
					System.out.println("MCV");
				} else {
					System.out.println("null");
				}
			} else {
				System.out.println();
				this.m_ADNodes[iValue].print();
			}
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
