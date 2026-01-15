"""
Example demonstrating the slice visualization API for CuTe layouts.

This example shows how to visualize which elements of a layout are
selected by a slice specification, with sliced elements highlighted
in dark grey.
"""

from cute_viz import render_layout_slice_svg
from cutlass import cute


@cute.jit
def main():
    # Create a 4x8 row-major layout
    layout = cute.make_layout((4, 8), stride=(8, 1))
    print(f"Layout: {layout}")
    print(f"Shape: (4, 8), Stride: (8, 1)")
    print()

    # Example 1: Highlight the first two rows
    print("Example 1: Slicing first two rows (slice(0, 2), None)")
    render_layout_slice_svg(layout, (slice(0, 2), None), "assets/slice_first_rows.svg")
    print("  -> Saved to assets/slice_first_rows.svg")

    # Example 2: Highlight a specific column
    print("Example 2: Slicing column 3 (None, 3)")
    render_layout_slice_svg(layout, (None, 3), "assets/slice_column.svg")
    print("  -> Saved to assets/slice_column.svg")

    # Example 3: Highlight a rectangular region
    print("Example 3: Slicing a rectangular region (slice(1, 3), slice(2, 6))")
    render_layout_slice_svg(layout, (slice(1, 3), slice(2, 6)), "assets/slice_rect.svg")
    print("  -> Saved to assets/slice_rect.svg")

    # Example 4: Highlight a single element
    print("Example 4: Slicing a single element (2, 4)")
    render_layout_slice_svg(layout, (2, 4), "assets/slice_single.svg")
    print("  -> Saved to assets/slice_single.svg")

    # Example 5: Select all (None)
    print("Example 5: Select all elements (None)")
    render_layout_slice_svg(layout, None, "assets/slice_all.svg")
    print("  -> Saved to assets/slice_all.svg")

    # Example 6: 1D layout slice
    print("\nExample 6: 1D layout with slice")
    layout_1d = cute.make_layout(8, stride=2)
    print(f"1D Layout: {layout_1d}")
    render_layout_slice_svg(layout_1d, slice(2, 6), "assets/slice_1d.svg")
    print("  -> Saved to assets/slice_1d.svg")

    # Repro the examples form CuTe docs
    # https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#slicing-a-tensor
    layout = cute.make_layout(((3, 2), (2, 5, 2)), stride=((4, 1), (2, 13, 100)))
    print(f"Layout: {layout}")

    # Example 7: Slice Row
    print("\nExample 7: T[2,:]")
    render_layout_slice_svg(layout, (2, None), "assets/slice_complex_row.svg")
    print("  -> Saved to assets/slice_complex_row.svg")

    # Example 8: Slice Column
    print("\nExample 8: T[:,5]")
    render_layout_slice_svg(layout, (None, 5), "assets/slice_complex_column.svg")
    print("  -> Saved to assets/slice_complex_column.svg")

    # Example 9: Complex Slice
    print("\nExample 9: T[(:,1),(0,:,1)]")
    render_layout_slice_svg(
        layout, ((None, 1), (0, None, 1)), "assets/slice_complex_3.svg"
    )
    print("  -> Saved to assets/slice_complex_3.svg")

    # Example 10: Complex Slice
    print("\nExample 10: T[(2,:),(:,3,:)]")
    render_layout_slice_svg(
        layout, ((2, None), (None, 3, None)), "assets/slice_complex_4.svg"
    )
    print("  -> Saved to assets/slice_complex_4.svg")

    print("\nDone! Check the assets/ directory for generated SVGs.")


if __name__ == "__main__":
    main()
