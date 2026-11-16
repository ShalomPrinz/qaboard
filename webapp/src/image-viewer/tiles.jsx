// With more than two images, each image besides the anchor shows in a tile: alone, wiped against the anchor, or as
// its difference with the anchor. A tile has its own viewers (comparison.js): its image, and the anchor when it's
// compared with it. Tiles follow the anchor through its sync group (sync.js), like the viewers of a page do.
import { useEffect, useEffectEvent, useLayoutEffect, useRef, useState, useSyncExternalStore } from 'react';

import { Comparison } from './comparison';
import { hasSource } from './grid';
import { Crosshair, DiffLayer, PixelValues, RegionOutlines } from './overlays';
import { Icon, Spinner } from './ui';

const clamp = (x, min, max) => Math.min(max, Math.max(min, x));

export const Tile = ({
  image,
  name,
  color,
  textColor,
  // what it's compared with
  anchor,
  anchorName,
  anchorSize,
  // 'side-by-side' and 'single': the image, 'wipe': anchor | image, 'difference': anchor − image
  mode,
  // another image shows in its place (single mode)
  hidden = false,
  // near the screen: else it doesn't follow the others nor compute the difference
  active = true,
  syncKey,
  left,
  top,
  width,
  height,
  filter,
  diff,
  wipe,
  onWipeChange,
  showLabel = true,
  // under the image's name, e.g. how far it is from the anchor
  note,
  regions,
  activeRegion,
  onSelectRegion,
  pixelSource = 'off',
  pixelTiles,
  base,
  // (event, controller): users clicked the image (not dragged it)
  onClick,
  // users want to compare the others with this image
  onAnchor,
  // (controller | null): for the values under the pointer, and how far apart the images are
  onController,
}) => {
  const [controller] = useState(() => new Comparison({ syncKey }));
  useEffect(() => () => controller.dispose(), [controller]);
  const snapshot = useSyncExternalStore(controller.subscribe, controller.getSnapshot);
  const available = hasSource(image);
  const compared = available && hasSource(anchor) && (mode === 'wipe' || mode === 'difference');

  useEffect(() => {
    controller.setImage('a', available ? image : null);
  });
  useEffect(() => {
    controller.setImage('b', compared ? anchor : null);
  });
  // the anchor's viewer goes with its pane
  useEffect(() => {
    if (!compared) controller.detach('b');
  }, [controller, compared]);
  useEffect(() => controller.setSyncKey(syncKey), [controller, syncKey]);
  const anchor_width = anchorSize?.width;
  const anchor_height = anchorSize?.height;
  useEffect(() => {
    controller.setGroupSize({ width: anchor_width, height: anchor_height });
  }, [controller, anchor_width, anchor_height]);
  useEffect(() => controller.setVisible(active), [controller, active]);
  const register = useEffectEvent(value => onController?.(value));
  useEffect(() => {
    register(controller);
    return () => register(null);
  }, [controller]);
  const clicked = useEffectEvent((event, element) => onClick?.(event, controller, element));
  useLayoutEffect(() => {
    controller.setOnClick((_pane, event, element) => clicked(event, element));
  }, [controller]);

  const [diff_error, setDiffError] = useState(null);
  const root = useRef(null);
  const wiping = useRef(false);
  const size = { width, height };
  const own_size = snapshot.a.width ? snapshot.a : null;
  const error = snapshot.a.error;
  const shown = active && !hidden;

  return <div
    ref={root}
    className={`qiv-tile qiv-tile-${mode}`}
    style={{ left, top, width, height, visibility: hidden ? 'hidden' : undefined, zIndex: hidden ? 0 : 1 }}
    onPointerMove={event => controller.hover('a', event.clientX, event.clientY)}
    onPointerLeave={() => controller.unhover()}
  >
    {compared && <div className="qiv-pane" style={{ ...size, left: 0, zIndex: 1, visibility: mode === 'difference' ? 'hidden' : undefined }}>
      <div className="qiv-osd" ref={element => { if (element) controller.attach('b', element); }} style={{ filter }} />
    </div>}
    <div className="qiv-pane" style={{ ...size, left: 0, zIndex: 2, clipPath: mode === 'wipe' && compared ? `inset(0 0 0 ${wipe * width}px)` : undefined }}>
      <div className="qiv-osd" ref={element => { if (element) controller.attach('a', element); }} style={{ filter }} />
      {available && (mode === 'side-by-side' || mode === 'single') && <PixelValues
        controller={controller}
        pane="a"
        order={['a', 'b']}
        source={shown ? pixelSource : 'off'}
        width={width}
        height={height}
        tiles={pixelTiles}
        images={{ a: image, b: null }}
        base={base}
      />}
    </div>
    {compared && mode === 'difference' && <div className="qiv-pane qiv-diff-pane" style={{ ...size, left: 0, zIndex: 3 }}>
      <DiffLayer controller={controller} order={['b', 'a']} diff={diff} active={shown} width={width} height={height} onError={setDiffError} />
      {diff_error && <div className="qiv-message qiv-pane-message">{diff_error}</div>}
    </div>}
    {shown && regions?.length > 0 && <RegionOutlines controller={controller} pane="a" regions={regions} active={activeRegion} size={anchorSize} paneSize={own_size} onSelect={onSelectRegion} />}
    <Crosshair controller={controller} pane="a" />
    {mode === 'wipe' && compared && <>
      <span className="qiv-label qiv-label-left qiv-label-quiet" title={`Left of the divider: ${anchorName}, the anchor`}>{anchorName}</span>
      <div
        className="qiv-wipe"
        style={{ left: wipe * width }}
        role="separator"
        aria-orientation="vertical"
        aria-label={`Wipe ${anchorName} | ${name}`}
        aria-valuenow={Math.round(100 * wipe)}
        aria-valuemin={0}
        aria-valuemax={100}
        tabIndex={0}
        onKeyDown={event => {
          const step = { ArrowLeft: -0.05, ArrowRight: 0.05 }[event.key];
          if (!step) return;
          event.preventDefault();
          onWipeChange(clamp(wipe + step, 0, 1));
        }}
        onPointerDown={event => {
          event.stopPropagation();
          event.currentTarget.setPointerCapture?.(event.pointerId);
          wiping.current = true;
        }}
        onPointerMove={event => {
          if (!wiping.current) return;
          const rect = root.current.getBoundingClientRect();
          onWipeChange(clamp((event.clientX - rect.left) / width, 0, 1));
        }}
        onPointerUp={() => { wiping.current = false; }}
        onPointerCancel={() => { wiping.current = false; }}
      >
        <div className="qiv-wipe-handle"><Icon icon="arrows-horizontal" size={14} /></div>
      </div>
    </>}
    {showLabel && <button
      type="button"
      className={`qiv-label ${mode === 'wipe' && compared ? 'qiv-label-right' : 'qiv-label-left'}`}
      style={{ background: color, color: textColor }}
      title={available ? `Make ${name} the anchor: the other images are compared with it` : undefined}
      disabled={!available}
      onClick={onAnchor}
      onPointerDown={event => event.stopPropagation()}
    >
      {name}{own_size && <span className="qiv-label-detail">{own_size.width}×{own_size.height}</span>}
    </button>}
    {note && available && !error && <span className="qiv-tile-note">{note}</span>}
    {!available && <div className="qiv-message qiv-pane-message qiv-tile-placeholder">
      {image?.loading ? <><Spinner label={`Loading ${name}`} /> <span>Loading {name}…</span></> : <span>{image?.missing ?? `No image for ${name}`}</span>}
    </div>}
    {error && <div className="qiv-message qiv-pane-message">
      <strong>Can't show the {name} image</strong>
      <span>{error.message}</span>
      {error.url && <a href={error.url} target="_blank" rel="noopener noreferrer">See why</a>}
    </div>}
  </div>;
};
